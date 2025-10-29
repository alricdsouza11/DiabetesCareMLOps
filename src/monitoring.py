import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import json
from pathlib import Path

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import *

class ModelMonitor:
    def __init__(self, reference_data: pd.DataFrame, model_version: str):
        """Initialize monitor with reference data (training set)"""
        self.reference_data = reference_data
        self.model_version = model_version
        self.reports_dir = Path("monitoring/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_data_quality_report(self, current_data: pd.DataFrame) -> Report:
        """Generate data quality report comparing current data to reference"""
        report = Report(metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
        ])
        report.run(reference_data=self.reference_data, current_data=current_data)
        return report
    
    def generate_target_drift_report(
        self, 
        current_data: pd.DataFrame, 
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Report:
        """Generate target drift report comparing predictions"""
        report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        ref_data = self.reference_data.copy()
        curr_data = current_data.copy()
        
        ref_data['prediction'] = reference_predictions
        curr_data['prediction'] = current_predictions
        
        report.run(reference_data=ref_data, current_data=curr_data, column_mapping={'target': 'prediction'})
        return report
    
    def save_report(self, report: Report, report_type: str):
        """Save Evidently report to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{self.model_version}_{timestamp}.html"
        report.save_html(self.reports_dir / filename)
        
    def calculate_prediction_drift_metrics(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate statistical metrics comparing prediction distributions"""
        from scipy import stats
        
        metrics = {}
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(reference_predictions, current_predictions)
        metrics['ks_statistic'] = ks_statistic
        metrics['ks_pvalue'] = ks_pvalue
        
        # Jensen-Shannon divergence
        def _jensen_shannon(p, q):
            p = np.asarray(p)
            q = np.asarray(q)
            p = p/np.sum(p)
            q = q/np.sum(q)
            m = (p + q) / 2
            return (stats.entropy(p, m) + stats.entropy(q, m)) / 2
            
        hist_ref, _ = np.histogram(reference_predictions, bins=20, density=True)
        hist_curr, _ = np.histogram(current_predictions, bins=20, density=True)
        js_div = _jensen_shannon(hist_ref, hist_curr)
        metrics['jensen_shannon_div'] = js_div
        
        return metrics
    
    def detect_data_drift(
        self, 
        current_data: pd.DataFrame,
        drift_threshold: float = 0.05
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Detect if there is significant data drift between reference and current data
        Returns:
            - bool: True if drift detected
            - dict: Drift metrics for each feature
        """
        drift_metrics = {}
        significant_drift = False
        
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in [np.number]:
                # For numeric columns, use Kolmogorov-Smirnov test
                statistic, pvalue = stats.ks_2samp(
                    self.reference_data[column].dropna(),
                    current_data[column].dropna()
                )
                drift_metrics[column] = {
                    'statistic': statistic,
                    'pvalue': pvalue,
                    'drift_detected': pvalue < drift_threshold
                }
                if pvalue < drift_threshold:
                    significant_drift = True
            else:
                # For categorical columns, use Chi-square test
                contingency = pd.crosstab(
                    pd.concat([self.reference_data[column], current_data[column]]),
                    pd.Series(['reference']*len(self.reference_data) + ['current']*len(current_data))
                )
                chi2, pvalue = stats.chi2_contingency(contingency)[:2]
                drift_metrics[column] = {
                    'statistic': chi2,
                    'pvalue': pvalue,
                    'drift_detected': pvalue < drift_threshold
                }
                if pvalue < drift_threshold:
                    significant_drift = True
        
        return significant_drift, drift_metrics
    
    def save_drift_metrics(self, metrics: Dict, prefix: str = "drift_metrics"):
        """Save drift metrics to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.model_version}_{timestamp}.json"
        
        with open(self.reports_dir / filename, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    def analyze_and_save_all(
        self,
        current_data: pd.DataFrame,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ):
        """Run all monitoring analyses and save reports"""
        # Data Quality and Drift Report
        quality_report = self.generate_data_quality_report(current_data)
        self.save_report(quality_report, "data_quality")
        
        # Target Drift Report
        target_report = self.generate_target_drift_report(
            current_data, reference_predictions, current_predictions
        )
        self.save_report(target_report, "target_drift")
        
        # Prediction Drift Metrics
        pred_metrics = self.calculate_prediction_drift_metrics(
            reference_predictions, current_predictions
        )
        self.save_drift_metrics(pred_metrics, "prediction_drift")
        
        # Feature Drift Detection
        drift_detected, drift_metrics = self.detect_data_drift(current_data)
        self.save_drift_metrics(drift_metrics, "feature_drift")
        
        return {
            'drift_detected': drift_detected,
            'prediction_metrics': pred_metrics,
            'drift_metrics': drift_metrics
        }