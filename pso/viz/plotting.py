import os
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(history: dict, title: str, filepath: str = None):
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(history['best_fitness'], linewidth=2, color='royalblue', label='Global Best')
    plt.yscale('log')
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Fitness (Escala Logarítmica)', fontsize=12)
    plt.legend()
    
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Gráfico de convergencia guardado en: {filepath}")
    
    plt.close()

def plot_speedup_bar(times: dict, title: str, filepath: str = None):
    
    labels = list(times.keys())
    values = list(times.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#e63946', '#f4a261', '#2a9d8f', '#264653', '#8ab17d'])
    
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Tiempo total (segundos)', fontsize=12)
    plt.xticks(rotation=15)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values)*0.01), 
                 f'{yval:.3f}s', ha='center', va='bottom', fontsize=10)
                 
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Gráfico de speedup guardado en: {filepath}")
        
    plt.close()