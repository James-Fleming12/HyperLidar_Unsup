import matplotlib.pyplot as plt
import json
import numpy as np
import os

SAVE_DIR = "logs/plots"
LOG_PATH = "logs/training_log.json"

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(LOG_PATH, 'r') as f:
        logs = json.load(f)
    
    epochs = logs['epochs']
    train_total = logs['train_total_loss']
    train_low = logs['train_low_loss']
    train_high = logs['train_high_loss']
    val_total = logs['val_total_loss']
    val_low = logs['val_low_loss']
    val_high = logs['val_high_loss']
    learning_rates = logs['learning_rates']

    pre_epochs = 0
    for i, high_loss in enumerate(train_high):
        if high_loss > 0:
            pre_epochs = i
            break
    
    print(f"Detected pre_epochs: {pre_epochs}")
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Total Loss (Training vs Validation)
    ax1.plot(epochs, train_total, 'b-', label='Train Total', linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_total, 'r-', label='Val Total', linewidth=2, alpha=0.8)
    ax1.axvline(x=pre_epochs, color='gray', linestyle='--', alpha=0.7, label='High Loss Start')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Components (Training)
    ax2.plot(epochs, train_total, 'k-', label='Train Total', linewidth=2, alpha=0.8)
    ax2.plot(epochs, train_low, 'b-', label='Train Low', linewidth=1.5, alpha=0.7)
    ax2.plot(epochs, train_high, 'g-', label='Train High', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=pre_epochs, color='gray', linestyle='--', alpha=0.7, label='High Loss Start')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Components (Validation)
    ax3.plot(epochs, val_total, 'k-', label='Val Total', linewidth=2, alpha=0.8)
    ax3.plot(epochs, val_low, 'b-', label='Val Low', linewidth=1.5, alpha=0.7)
    ax3.plot(epochs, val_high, 'g-', label='Val High', linewidth=1.5, alpha=0.7)
    ax3.axvline(x=pre_epochs, color='gray', linestyle='--', alpha=0.7, label='High Loss Start')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Validation Loss Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate and High/Low Ratio
    ax4_twin = ax4.twinx()
    
    # Learning rate (left axis)
    ax4.semilogy(epochs, learning_rates, 'purple', label='Learning Rate', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate', color='purple')
    ax4.tick_params(axis='y', labelcolor='purple')
    
    # High/Low ratio (right axis) - only after pre_epochs
    high_epochs = [e for e in epochs if e >= pre_epochs]
    train_ratios = [h/l if l > 0 else 0 for h, l in zip(train_high[pre_epochs:], train_low[pre_epochs:])]
    val_ratios = [h/l if l > 0 else 0 for h, l in zip(val_high[pre_epochs:], val_low[pre_epochs:])]
    
    ax4_twin.plot(high_epochs, train_ratios, 'orange', label='Train High/Low Ratio', linewidth=1.5, alpha=0.8)
    ax4_twin.plot(high_epochs, val_ratios, 'red', label='Val High/Low Ratio', linewidth=1.5, alpha=0.8)
    ax4_twin.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal Ratio')
    ax4_twin.set_ylabel('High/Low Ratio', color='red')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    ax4.set_title('Learning Rate and Loss Ratio')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends for the fourth plot
    lines4, labels4 = ax4.get_legend_handles_labels()
    lines4_twin, labels4_twin = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines4 + lines4_twin, labels4 + labels4_twin, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/training_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create individual detailed plots
    create_individual_plots(logs, pre_epochs, SAVE_DIR)
    
    # Create convergence analysis
    create_convergence_analysis(logs, SAVE_DIR)

def create_individual_plots(logs, pre_epochs, save_dir):
    """Create individual detailed plots"""
    epochs = logs['epochs']
    
    # Plot 1: Just the validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, logs['val_total_loss'], 'k-', label='Val Total', linewidth=2)
    plt.plot(epochs, logs['val_low_loss'], 'b-', label='Val Low', linewidth=1.5, alpha=0.8)
    plt.plot(epochs, logs['val_high_loss'], 'g-', label='Val High', linewidth=1.5, alpha=0.8)
    plt.axvline(x=pre_epochs, color='gray', linestyle='--', alpha=0.7, label='High Loss Start')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/validation_losses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Training vs Validation comparison
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, logs['train_total_loss'], 'b-', label='Train Total', linewidth=2, alpha=0.8)
    plt.plot(epochs, logs['val_total_loss'], 'r-', label='Val Total', linewidth=2, alpha=0.8)
    plt.axvline(x=pre_epochs, color='gray', linestyle='--', alpha=0.7, label='High Loss Start')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/train_vs_val.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, logs['learning_rates'], 'purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/learning_rate.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_convergence_analysis(logs, save_dir):
    """Analyze and plot convergence metrics"""
    epochs = logs['epochs']
    val_loss = logs['val_total_loss']
    
    # Find best epoch
    best_epoch = np.argmin(val_loss)
    best_loss = val_loss[best_epoch]
    
    print(f"Best validation loss: {best_loss:.4f} at epoch {best_epoch}")
    
    # Plot convergence
    plt.figure(figsize=(12, 8))
    
    # Main convergence plot
    plt.subplot(2, 1, 1)
    plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.8, label=f'Best Epoch: {best_epoch}')
    plt.axhline(y=best_loss, color='green', linestyle='--', alpha=0.5, label=f'Best Loss: {best_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Model Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoomed in view of last 20% of training
    plt.subplot(2, 1, 2)
    zoom_start = int(len(epochs) * 0.8)
    zoom_epochs = epochs[zoom_start:]
    zoom_loss = val_loss[zoom_start:]
    
    plt.plot(zoom_epochs, zoom_loss, 'r-', linewidth=2)
    plt.axhline(y=best_loss, color='green', linestyle='--', alpha=0.5, label=f'Best Loss: {best_loss:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Final Convergence (Last 20% of Training)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()