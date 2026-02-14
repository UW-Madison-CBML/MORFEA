#!/usr/bin/env python3
"""
繪製訓練過程中的 loss 曲線圖
支持從 JSON 日誌或文本日誌中讀取數據
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re

def load_json_log(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    epochs = [entry['epoch'] for entry in data]
    total_loss = [entry['total'] for entry in data]
    reconstruction_loss = [entry['reconstruction'] for entry in data]
    l1_loss = [entry.get('l1', 0) for entry in data]
    ms_ssim_loss = [entry.get('ms_ssim', 0) for entry in data]
    smooth_loss = [entry.get('smooth', 0) for entry in data]
    learning_rate = [entry.get('lr', 0) for entry in data]
    
    return {
        'epochs': epochs,
        'total': total_loss,
        'reconstruction': reconstruction_loss,
        'l1': l1_loss,
        'ms_ssim': ms_ssim_loss,
        'smooth': smooth_loss,
        'lr': learning_rate
    }

def load_text_log(text_path):
    epochs = []
    losses = []
    
    with open(text_path, 'r') as f:
        for line in f:
            match = re.search(r'epoch (\d+) avg loss=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)
    
    return {
        'epochs': epochs,
        'total': losses,
        'reconstruction': None,
        'l1': None,
        'ms_ssim': None,
        'smooth': None,
        'lr': None
    }

def plot_loss_curves(data, output_path, show_components=True, show_lr=False):
    epochs = data['epochs']
    total_loss = data['total']
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    
    axes_twin = None
    
    axes.plot(epochs, total_loss, color='#1f77b4', linewidth=2.5, 
             label='Total Loss', marker='o', markersize=5, markevery=5)
    
    if show_components and data['reconstruction'] is not None:
        axes.plot(epochs, data['reconstruction'], color='#2ca02c', 
                 linestyle='--', linewidth=2, label='Reconstruction Loss')
        
        if data['l1'] is not None and any(v > 0 for v in data['l1']):
            axes.plot(epochs, data['l1'], color='#d62728', 
                     linestyle=':', linewidth=2, label='L1 Loss (MAE)')
        
        if data['ms_ssim'] is not None and any(v > 0 for v in data['ms_ssim']):
            axes.plot(epochs, data['ms_ssim'], color='#9467bd', 
                     linestyle='-.', linewidth=2, label='MS-SSIM Loss')
        
        if data['smooth'] is not None and any(v > 0 for v in data['smooth']):
            smooth_scaled = [v * 1000 for v in data['smooth']]
            axes_twin = axes.twinx()
            axes_twin.plot(epochs, smooth_scaled, color='#ff7f0e', 
                          linewidth=2.5, label='Smooth Loss (×1000)', marker='s', markersize=4, markevery=5)
            axes_twin.set_ylabel('Smooth Loss (scaled)', color='#ff7f0e', fontsize=11, fontweight='bold')
            axes_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    
    axes.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes.set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes.set_title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
    axes.grid(True, alpha=0.3, linestyle='--')
    
    min_loss = min(total_loss)
    min_epoch = epochs[total_loss.index(min_loss)]
    final_loss = total_loss[-1]
    
    axes.annotate(f'Min: {min_loss:.4f} @ Epoch {min_epoch}',
                 xy=(min_epoch, min_loss), xytext=(10, 10),
                 textcoords='offset points', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                          edgecolor='black', linewidth=1, alpha=0.9),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    axes.text(0.70, 0.98, f'Final Loss: {final_loss:.4f}\n'
                          f'Total Epochs: {len(epochs)}\n'
                          f'Improvement: {((total_loss[0] - final_loss) / total_loss[0] * 100):.1f}%',
             transform=axes.transAxes, fontsize=9,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor='black', linewidth=1, alpha=0.9))
    
    if axes_twin is not None:
        lines1, labels1 = axes.get_legend_handles_labels()
        lines2, labels2 = axes_twin.get_legend_handles_labels()
        axes.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
                   fontsize=10, framealpha=0.9)
    else:
        axes.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_lr and data['lr'] is not None and any(v > 0 for v in data['lr']):
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        ax2.plot(epochs, data['lr'], 'orange', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        lr_output = output_path.replace('.png', '_lr.png')
        plt.savefig(lr_output, dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='繪製訓練 loss 曲線圖')
    parser.add_argument('--json_log', type=str, 
                       default='logs/training_log.json',
                       help='JSON 格式的訓練日誌路徑')
    parser.add_argument('--text_log', type=str,
                       help='文本格式的訓練日誌路徑（如果沒有 JSON）')
    parser.add_argument('--output', type=str,
                       default='training_loss_curve.png',
                       help='輸出圖片路徑')
    parser.add_argument('--no_components', action='store_true',
                       help='不顯示詳細的 loss 組件')
    parser.add_argument('--show_lr', action='store_true',
                       help='同時繪製 learning rate 曲線')
    
    args = parser.parse_args()
    
    data = None
    
    if Path(args.json_log).exists():
        data = load_json_log(args.json_log)
    elif args.text_log and Path(args.text_log).exists():
        data = load_text_log(args.text_log)
    else:
        possible_json = Path('logs/training_log.json')
        possible_text = Path('../training_log.txt')
        
        if possible_json.exists():
            data = load_json_log(possible_json)
        elif possible_text.exists():
            data = load_text_log(possible_text)
        else:
            return
    
    if data is None or len(data['epochs']) == 0:
        return
    
    
    plot_loss_curves(data, args.output, 
                     show_components=not args.no_components,
                     show_lr=args.show_lr)
    

if __name__ == "__main__":
    main()

