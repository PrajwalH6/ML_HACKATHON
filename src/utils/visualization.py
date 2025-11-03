"""Plotting utilities for analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_style("whitegrid")

def plot_training_curve(rewards, save_path='reports/figures/training_curve.png'):
    """Plot training reward curve."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Smooth curve
    window = min(100, len(rewards) // 10)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label='Smoothed Reward', linewidth=2)
    
    plt.plot(rewards, alpha=0.3, label='Raw Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('RL Training Progress')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curve to {save_path}")


def plot_evaluation_results(metrics, df, save_dir='reports/figures'):
    """Plot evaluation results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Success rate
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Win/Loss pie chart
    axes[0, 0].pie([metrics['wins'], metrics['losses']], 
                   labels=['Wins', 'Losses'],
                   autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    
    # Wrong guesses distribution
    axes[0, 1].hist(df['wrong_guesses'], bins=range(0, 8), edgecolor='black')
    axes[0, 1].set_xlabel('Wrong Guesses')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Wrong Guesses Distribution')
    
    # Repeated guesses distribution
    axes[1, 0].hist(df['repeated_guesses'], bins=range(0, max(df['repeated_guesses'])+2), edgecolor='black')
    axes[1, 0].set_xlabel('Repeated Guesses')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Repeated Guesses Distribution')
    
    # Performance by word length
    df['word_length'] = df['word'].apply(len)
    length_stats = df.groupby('word_length')['won'].agg(['sum', 'count'])
    length_stats['success_rate'] = length_stats['sum'] / length_stats['count']
    axes[1, 1].bar(length_stats.index, length_stats['success_rate'])
    axes[1, 1].set_xlabel('Word Length')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].set_title('Success Rate by Word Length')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'evaluation_summary.png', dpi=300)
    plt.close()
    
    print(f"Saved evaluation plots to {save_dir}")
