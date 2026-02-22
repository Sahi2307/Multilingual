"""Visualize training metrics and performance dashboard.

This script generates comprehensive visualizations for both MuRIL category
classifier and XGBoost urgency predictor, including:
- Training/validation accuracy curves
- Loss curves over epochs
- Confusion matrices
- F1-scores per class
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


def plot_category_model_metrics(models_dir: Path, output_dir: Path):
    """Plot MuRIL category classifier training metrics."""
    logger.info("Plotting category model metrics...")
    
    # Load training history
    history_path = models_dir / "muril_category_classifier" / "training_history.json"
    if not history_path.exists():
        logger.warning(f"Training history not found at {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Extract metrics
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_accuracy = [h['val_accuracy'] for h in history]
    val_f1 = [h['val_f1'] for h in history]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training and Validation Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('MuRIL Category Classifier - Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, val_accuracy, 'g-^', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.axhline(y=0.9333, color='orange', linestyle='--', linewidth=2, label='Target (93.33%)')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('MuRIL Category Classifier - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Plot 3: Validation F1-Score
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, val_f1, 'm-D', label='Validation F1-Score', linewidth=2, markersize=6)
    ax3.axhline(y=0.9333, color='orange', linestyle='--', linewidth=2, label='Target (93.33%)')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('MuRIL Category Classifier - F1-Score', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # Plot 4: Summary Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    best_epoch = epochs[np.argmax(val_accuracy)]
    best_accuracy = max(val_accuracy)
    best_f1 = val_f1[np.argmax(val_accuracy)]
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    
    summary_text = f"""
    📊 MuRIL Category Classifier Summary
    {'='*45}
    
    Best Performance:
    • Best Epoch: {best_epoch}
    • Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
    • Best F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%)
    
    Final Metrics:
    • Final Train Loss: {final_train_loss:.4f}
    • Final Val Loss: {final_val_loss:.4f}
    • Total Epochs: {len(epochs)}
    
    Target Achievement:
    • Accuracy Target (≥93.33%): {'✓ ACHIEVED' if best_accuracy >= 0.9333 else '✗ NOT MET'}
    • F1-Score Target (≥93.33%): {'✓ ACHIEVED' if best_f1 >= 0.9333 else '✗ NOT MET'}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.suptitle('MuRIL Category Classifier - Training Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = output_dir / "category_model_training_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved category model metrics to {output_path}")
    plt.close()


def plot_urgency_model_metrics(models_dir: Path, output_dir: Path):
    """Plot XGBoost urgency predictor metrics."""
    logger.info("Plotting urgency model metrics...")
    
    # Load test results
    results_path = models_dir / "urgency_test_results.json"
    if not results_path.exists():
        logger.warning(f"Test results not found at {results_path}")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array(results['confusion_matrix'])
    urgency_levels = ['Critical', 'High', 'Medium', 'Low']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=urgency_levels,
                yticklabels=urgency_levels, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_title('XGBoost Urgency Predictor - Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Plot 2: Per-Class Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    per_class_acc = results['per_class_accuracy']
    classes = list(per_class_acc.keys())
    accuracies = [per_class_acc[c] for c in classes]
    
    colors = ['#ff6b6b', '#f06595', '#cc5de8', '#845ef7']
    bars = ax2.bar(classes, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=0.89, color='orange', linestyle='--', linewidth=2, label='Target (89%)')
    ax2.set_xlabel('Urgency Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('XGBoost Urgency Predictor - Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Overall Metrics
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
    
    colors_metrics = ['#51cf66', '#339af0', '#ff6b6b', '#ffd43b']
    bars = ax3.barh(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.axvline(x=0.89, color='orange', linestyle='--', linewidth=2, label='Target (89%)')
    ax3.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('XGBoost Urgency Predictor - Overall Metrics', fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 1.05])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax3.text(val, bar.get_y() + bar.get_height()/2.,
                f' {val:.4f} ({val*100:.2f}%)', va='center', fontweight='bold')
    
    # Plot 4: Summary Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    summary_text = f"""
    📊 XGBoost Urgency Predictor Summary
    {'='*45}
    
    Overall Performance:
    • Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
    • Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)
    • Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)
    • F1-Score: {results['f1']:.4f} ({results['f1']*100:.2f}%)
    
    Per-Class Accuracy:
    • Critical: {per_class_acc.get('Critical', 0):.4f} ({per_class_acc.get('Critical', 0)*100:.2f}%)
    • High: {per_class_acc.get('High', 0):.4f} ({per_class_acc.get('High', 0)*100:.2f}%)
    • Medium: {per_class_acc.get('Medium', 0):.4f} ({per_class_acc.get('Medium', 0)*100:.2f}%)
    • Low: {per_class_acc.get('Low', 0):.4f} ({per_class_acc.get('Low', 0)*100:.2f}%)
    
    Target Achievement:
    • Accuracy Target (≥89%): {'✓ ACHIEVED' if results['accuracy'] >= 0.89 else '✗ NOT MET'}
    • F1-Score Target (≥88%): {'✓ ACHIEVED' if results['f1'] >= 0.88 else '✗ NOT MET'}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('XGBoost Urgency Predictor - Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = output_dir / "urgency_model_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved urgency model metrics to {output_path}")
    plt.close()


def create_combined_dashboard(models_dir: Path, output_dir: Path):
    """Create a combined performance dashboard for both models."""
    logger.info("Creating combined performance dashboard...")
    
    # Load both results
    category_history_path = models_dir / "muril_category_classifier" / "training_history.json"
    urgency_results_path = models_dir / "urgency_test_results.json"
    
    if not category_history_path.exists() or not urgency_results_path.exists():
        logger.warning("Missing required files for combined dashboard")
        return
    
    with open(category_history_path, 'r') as f:
        category_history = json.load(f)
    
    with open(urgency_results_path, 'r') as f:
        urgency_results = json.load(f)
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Category Model - Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = [h['epoch'] for h in category_history]
    val_accuracy = [h['val_accuracy'] for h in category_history]
    ax1.plot(epochs, val_accuracy, 'g-^', linewidth=2.5, markersize=7)
    ax1.axhline(y=0.9333, color='orange', linestyle='--', linewidth=2, label='Target (93.33%)')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('MuRIL - Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Category Model - Loss
    ax2 = fig.add_subplot(gs[0, 1])
    train_loss = [h['train_loss'] for h in category_history]
    val_loss = [h['val_loss'] for h in category_history]
    ax2.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax2.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('MuRIL - Loss Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Category Model - F1-Score
    ax3 = fig.add_subplot(gs[0, 2])
    val_f1 = [h['val_f1'] for h in category_history]
    ax3.plot(epochs, val_f1, 'm-D', linewidth=2.5, markersize=7)
    ax3.axhline(y=0.9333, color='orange', linestyle='--', linewidth=2, label='Target (93.33%)')
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax3.set_title('MuRIL - F1-Score', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # Urgency Model - Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm = np.array(urgency_results['confusion_matrix'])
    urgency_levels = ['Critical', 'High', 'Medium', 'Low']
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=urgency_levels,
                yticklabels=urgency_levels, ax=ax4, cbar_kws={'label': 'Count'})
    ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax4.set_ylabel('True', fontsize=11, fontweight='bold')
    ax4.set_title('XGBoost - Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Urgency Model - Metrics
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [urgency_results['accuracy'], urgency_results['precision'], 
              urgency_results['recall'], urgency_results['f1']]
    colors = ['#51cf66', '#339af0', '#ff6b6b', '#ffd43b']
    bars = ax5.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.axvline(x=0.89, color='orange', linestyle='--', linewidth=2, label='Target')
    ax5.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax5.set_title('XGBoost - Overall Metrics', fontsize=12, fontweight='bold')
    ax5.set_xlim([0, 1.05])
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='x')
    for bar, val in zip(bars, values):
        ax5.text(val, bar.get_y() + bar.get_height()/2., f' {val:.3f}', 
                va='center', fontweight='bold', fontsize=9)
    
    # Urgency Model - Per-Class Accuracy
    ax6 = fig.add_subplot(gs[1, 2])
    per_class_acc = urgency_results['per_class_accuracy']
    classes = list(per_class_acc.keys())
    accuracies = [per_class_acc[c] for c in classes]
    colors_class = ['#ff6b6b', '#f06595', '#cc5de8', '#845ef7']
    bars = ax6.bar(classes, accuracies, color=colors_class, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.axhline(y=0.89, color='orange', linestyle='--', linewidth=2, label='Target')
    ax6.set_xlabel('Urgency Level', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax6.set_title('XGBoost - Per-Class Accuracy', fontsize=12, fontweight='bold')
    ax6.set_ylim([0, 1.05])
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle('Civic Complaint System - ML Models Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = output_dir / "combined_performance_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved combined dashboard to {output_path}")
    plt.close()


def main():
    """Generate all performance visualizations."""
    root_dir = Path(__file__).resolve().parents[1]
    models_dir = root_dir / "models"
    output_dir = root_dir / "data" / "visualizations"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting visualization generation...")
    
    try:
        # Generate individual dashboards
        plot_category_model_metrics(models_dir, output_dir)
        plot_urgency_model_metrics(models_dir, output_dir)
        
        # Generate combined dashboard
        create_combined_dashboard(models_dir, output_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info("All visualizations generated successfully!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"{'='*60}\n")
    except Exception as e:
        logger.exception(f"Fatal error in visualization generation: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
