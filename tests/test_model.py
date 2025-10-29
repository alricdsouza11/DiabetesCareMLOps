import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.data_preprocessing import build_preprocessor, enrich_and_clean
from src.model_training import train_model_with_cv
from src.monitoring import ModelMonitor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.choice(['[0-10)', '[30-40)', '[60-70)', '[90-100)'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Other'], n_samples),
        'time_in_hospital': np.random.randint(1, 14, n_samples),
        'num_lab_procedures': np.random.randint(1, 100, n_samples),
        'num_procedures': np.random.randint(0, 6, n_samples),
        'num_medications': np.random.randint(1, 50, n_samples),
        'number_outpatient': np.random.randint(0, 10, n_samples),
        'number_emergency': np.random.randint(0, 10, n_samples),
        'number_inpatient': np.random.randint(0, 10, n_samples),
        'number_diagnoses': np.random.randint(1, 16, n_samples),
        'max_glu_serum': np.random.choice(['None', 'Norm', '>200', '>300'], n_samples),
        'A1Cresult': np.random.choice(['None', 'Norm', '>7', '>8'], n_samples),
        'insulin': np.random.choice(['No', 'Steady', 'Up', 'Down'], n_samples),
        'readmitted_30': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% readmission rate
    }
    return pd.DataFrame(data)

def test_preprocessor_build(sample_data):
    """Test preprocessor building"""
    pre, num_cols, cat_cols = build_preprocessor(sample_data, 'readmitted_30')
    assert isinstance(pre, Pipeline)
    assert len(num_cols) > 0
    assert len(cat_cols) > 0
    assert all(col in sample_data.columns for col in num_cols + cat_cols)

def test_data_enrichment(sample_data):
    """Test data enrichment function"""
    enriched_df = enrich_and_clean(sample_data)
    assert isinstance(enriched_df, pd.DataFrame)
    assert len(enriched_df) == len(sample_data)
    assert 'readmitted_30' in enriched_df.columns

def test_model_training(sample_data):
    """Test model training with cross-validation"""
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1
    }
    
    pipe, cv_scores = train_model_with_cv(sample_data, 'readmitted_30', params)
    
    assert isinstance(pipe, Pipeline)
    assert len(cv_scores) == 5  # 5-fold CV
    assert all(0 <= score <= 1 for score in cv_scores)

def test_model_monitor(sample_data):
    """Test model monitoring functionality"""
    # Split data into reference and current
    split_idx = len(sample_data) // 2
    reference_data = sample_data.iloc[:split_idx].copy()
    current_data = sample_data.iloc[split_idx:].copy()
    
    # Create monitor
    monitor = ModelMonitor(reference_data, model_version='test')
    
    # Test drift detection
    drift_detected, metrics = monitor.detect_data_drift(current_data)
    assert isinstance(drift_detected, bool)
    assert isinstance(metrics, dict)
    assert len(metrics) == len(reference_data.columns)
    
    # Test prediction drift metrics
    ref_preds = np.random.random(len(reference_data))
    curr_preds = np.random.random(len(current_data))
    pred_metrics = monitor.calculate_prediction_drift_metrics(ref_preds, curr_preds)
    
    assert isinstance(pred_metrics, dict)
    assert 'ks_statistic' in pred_metrics
    assert 'jensen_shannon_div' in pred_metrics