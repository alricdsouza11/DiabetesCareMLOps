import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, precision_recall_curve, auc
import optuna
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

from src.data_preprocessing import build_preprocessor, enrich_and_clean
from src.evaluate import compute_metrics

def objective(trial, X, y, pre, num_cols, cat_cols):
    """Optuna objective for hyperparameter optimization"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
    }
    
    clf = XGBClassifier(
        **params,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    
    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use both ROC-AUC and PR-AUC for evaluation
    scores_roc = cross_val_score(pipe, X, y, scoring='roc_auc', cv=cv)
    scores_pr = cross_val_score(
        pipe, X, y,
        scoring=make_scorer(lambda y_true, y_pred: auc(*precision_recall_curve(y_true, y_pred)[:2])),
        cv=cv
    )
    
    # Combine metrics (weighted average favoring PR-AUC for imbalanced data)
    final_score = 0.4 * scores_roc.mean() + 0.6 * scores_pr.mean()
    
    return final_score

def train_with_optuna(df: pd.DataFrame, target_col: str, n_trials: int = 100):
    """Train model with Optuna hyperparameter optimization"""
    mlflow.set_experiment("hospital-readmission-optimized")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # Build preprocessor once for all trials
    pre, num_cols, cat_cols = build_preprocessor(df, target_col)
    
    # Create study object for maximization
    study = optuna.create_study(direction='maximize')
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X, y, pre, num_cols, cat_cols),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    
    with mlflow.start_run(run_name="xgb_optimized") as run:
        # Log study results
        mlflow.log_params(best_params)
        mlflow.log_metric("best_score", study.best_value)
        
        # Train final model with best parameters
        best_clf = XGBClassifier(
            **best_params,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        final_pipe = Pipeline([("pre", pre), ("clf", best_clf)])
        
        # Fit on full training data
        final_pipe.fit(X, y)
        
        # Get predictions
        y_prob = final_pipe.predict_proba(X)[:,1]
        
        # Log metrics
        metrics = compute_metrics(y, y_prob)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        # Log feature importances
        feature_names = (
            num_cols + 
            [f"{col}_{val}" for col, vals in 
             zip(cat_cols, pre.named_transformers_['cat'].named_steps['onehot'].categories_)
             for val in vals]
        )
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': best_clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save feature importances plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances['importance'])
        plt.xticks(range(len(importances)), importances['feature'], rotation=90)
        plt.title('Feature Importances')
        plt.tight_layout()
        mlflow.log_figure(plt.gcf(), "feature_importances.png")
        
        # Save model
        mlflow.sklearn.log_model(
            final_pipe,
            "model",
            registered_model_name="hospital_readmission_optimized"
        )
        
        return final_pipe, metrics, importances

def train_model_with_cv(df: pd.DataFrame, target_col: str, params: dict):
    """Train model with cross-validation for a specific parameter set"""
    mlflow.set_experiment("hospital-readmission-cv")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    pre, num_cols, cat_cols = build_preprocessor(df, target_col)
    
    with mlflow.start_run(run_name="xgb_cv") as run:
        clf = XGBClassifier(**params, eval_metric='logloss', random_state=42)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        
        # Perform cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X, y, scoring='roc_auc', cv=cv)
        
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        
        # Train final model on full dataset
        pipe.fit(X, y)
        
        return pipe, cv_scores