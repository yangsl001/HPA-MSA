# src/utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from scipy.stats import pearsonr


def calculate_metrics(preds_reg, truths_reg, preds_cls, truths_cls,
                      use_regr_for_class=False):
    """
    Calculates all required metrics for the HPA-MSA model.
    Can also compute classification metrics based on regression outputs.

    Args:
        preds_reg (np.array): Predictions from the regression head.
        truths_reg (np.array): Ground truth for the regression task.
        preds_cls (np.array): Predictions from the classification head (class indices).
        truths_cls (np.array): Ground truth for the classification task.
        use_regr_for_class (bool): If True, computes additional classification metrics
                                   based on the sign of regression outputs.

    Returns:
        dict: A dictionary containing all computed metrics.
    """
    # Ensure inputs are numpy arrays
    preds_reg = np.array(preds_reg)
    truths_reg = np.array(truths_reg)
    preds_cls = np.array(preds_cls)
    truths_cls = np.array(truths_cls)

    # --- 1. Standard Regression Metrics ---
    mae = mean_absolute_error(truths_reg, preds_reg)
    # pearsonr returns (correlation, p-value)
    # Add a small epsilon to avoid error if standard deviation is zero
    corr, _ = pearsonr(truths_reg.flatten(), preds_reg.flatten())

    # --- 2. Standard Classification Metrics (from classification head) ---
    acc = accuracy_score(truths_cls, preds_cls)
    f1 = f1_score(truths_cls, preds_cls, average='weighted')

    metrics = {
        'MAE': mae,
        'Corr': corr,
        'Acc_cls_head': acc,  # Renamed for clarity
        'F1_cls_head': f1,  # Renamed for clarity
    }

    # --- 3. (NEW) Classification Metrics based on Regression Outputs ---
    if use_regr_for_class:
        # Rule: > 1 is positive (class 1), <= 1 is non-positive (class 0)
        regr_preds_as_cls = (preds_reg > 1).astype(int)
        regr_truths_as_cls = (truths_reg > 1).astype(int)

        acc_regr = accuracy_score(regr_truths_as_cls, regr_preds_as_cls)
        f1_regr = f1_score(regr_truths_as_cls, regr_preds_as_cls, average='weighted')

        metrics['Acc_regr_head'] = acc_regr
        metrics['F1_regr_head'] = f1_regr

    return metrics


def calculate_metrics_reg(predictions, truths):
    """
    Calculates a standard set of 4 metrics for multimodal sentiment analysis.
    All classification metrics are derived from the regression outputs.

    Args:
        predictions (np.array): A numpy array of regression predictions.
        truths (np.array): A numpy array of ground truth regression labels.

    Returns:
        dict: A dictionary containing 'MAE', 'Corr', 'Accuracy', and 'F1-Score'.
    """
    # --- 1. Regression Metrics ---
    mae = mean_absolute_error(truths, predictions)

    # Pearson Correlation requires 1D arrays
    # Add a small epsilon to standard deviation to avoid NaN for constant inputs
    if np.std(truths) < 1e-6 or np.std(predictions) < 1e-6:
        corr = 0.0  # Cannot compute correlation if one array is constant
    else:
        corr, _ = pearsonr(truths.flatten(), predictions.flatten())

    # --- 2. Classification Metrics (derived from regression) ---
    # Rule: > 0 is positive (class 1), <= 0 is non-positive (class 0)
    preds_binary = (predictions > 1).astype(int)
    truths_binary = (truths > 1).astype(int)

    acc = accuracy_score(truths_binary, preds_binary)
    # Use 'weighted' average for F1-score to handle potential class imbalance
    f1 = f1_score(truths_binary, preds_binary, average='weighted')

    return {
        'MAE': mae,
        'Corr': corr,
        'Accuracy': acc,
        'F1-Score': f1,  # This will be our primary metric for model selection
    }