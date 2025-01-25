from typing import Dict, Any
import numpy as np
import warnings
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities

    Returns:
        Dictionary with various metrics
    """
    # Basic metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Advanced metrics with warning handling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except:
            roc_auc = 0.0

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision) if len(precision) > 1 else 0.0

    # Calculate other metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2

    return {
        'confusion_matrix': conf_matrix,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision': precision,
        'npv': npv,
        'f1_score': f1,
        'balanced_accuracy': balanced_accuracy,
        'classification_report': classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )
    }


def print_metrics(metrics: Dict[str, Any], technique: str = None) -> None:
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        technique: Optional name of propaganda technique
    """
    if technique:
        print(f"\nMetrics for {technique}:")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"\nROC AUC: {metrics['roc_auc']:.3f}")
    print(f"PR AUC: {metrics['pr_auc']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"NPV: {metrics['npv']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
