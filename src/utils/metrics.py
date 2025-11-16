
"""
Metrics Module for HazardLoc
Comprehensive evaluation metrics for classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Compute comprehensive classification metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Optional probability scores for ROC AUC.
                For binary classification, this should be the probability of the positive class.
                For multi-class, this should be a shape (n_samples, n_classes) array.
        class_names: Optional class names for reporting

    Returns:
        Dictionary of metrics
    """
    # --- FIX: Ensure inputs are NumPy arrays for consistent handling ---
    # This prevents the AttributeError when trying to access .shape on a list.
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print(f"--- Debugging compute_metrics ---")
    print(f"y_true unique values: {np.unique(y_true)}")
    print(f"y_pred unique values: {np.unique(y_pred)}")
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

    raw_accuracy = accuracy_score(y_true, y_pred)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Raw Accuracy (before *100): {raw_accuracy:.6f}")
    print(f"Weighted Recall: {weighted_recall:.6f}")
    print(f"Are they numerically equal? {np.isclose(raw_accuracy, weighted_recall)}")
    print(f"---------------------------------")

    # Note: 'recall' is weighted recall, which is mathematically equivalent to accuracy.
    metrics = {
        'accuracy': raw_accuracy * 100,
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': weighted_recall,
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # Add macro-averaged recall for a different perspective on class balance
    metrics['macro_recall'] = macro_recall

    # --- NEW: Add ROC AUC Score ---
    # This requires probability scores.
    if y_prob is not None:
        num_classes = len(np.unique(y_true))
        if num_classes > 2:
            # Multi-class case: One-vs-Rest strategy
            # y_prob should be shape (n_samples, n_classes)
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        elif num_classes == 2:
            # Binary case: y_prob should be the probability of the positive class (class 1)
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics['roc_auc'] = 0.0 # Not applicable for a single class

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    if class_names:
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        metrics['per_class'] = report

    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation curves
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history['val_acc'], 'g-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")

    plt.show()


def print_metrics_report(metrics, class_names=None):
    """
    Print formatted metrics report
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall (Weighted): {metrics['recall']:.4f}")
    print(f"Macro Recall: {metrics.get('macro_recall', 'N/A'):.4f}") # Use .get for safe access
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    if 'per_class' in metrics and class_names:
        print("\nPer-Class Metrics:")
        print("-"*60)
        for cls in class_names:
            cls_metrics = metrics['per_class'][cls]
            print(f"{cls}:")
            print(f"  Precision: {cls_metrics['precision']:.4f}")
            print(f"  Recall:    {cls_metrics['recall']:.4f}")
            print(f"  F1 Score:  {cls_metrics['f1-score']:.4f}")
            print(f"  Support:   {cls_metrics['support']}")

    print("="*60 + "\n")