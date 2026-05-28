import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pso.core.engine import PSOEngine
from pso.objectives.benchmarks import Ackley
from pso.parallel.evaluators import VectorizedEvaluator

def generate_report_visuals():
    dim = 30
    max_iterations = 100
    seeds = [42, 100, 2026, 9999, 12345] 
    benchmark = Ackley(dimensions=dim)
    evaluator = VectorizedEvaluator()

    histories = []
    final_fitnesses = []

    print(f"Ejecutando PSO {len(seeds)} veces para generar estadísticas...")

    for s in seeds:
        engine = PSOEngine(
            num_particles=50, dimensions=dim, bounds=benchmark.bounds,
            evaluator=evaluator, seed=s,
            w=0.729, c1=1.49445, c2=1.49445 
        )
        res = engine.optimize(benchmark, max_iterations=max_iterations)
        histories.append(res['history']['best_fitness'])
        final_fitnesses.append(res['gbest_fitness'])

    histories = np.array(histories) 
    mean_curve = np.mean(histories, axis=0)
    std_curve = np.std(histories, axis=0)
    iterations = np.arange(len(mean_curve))

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, mean_curve, 'b-', linewidth=2, label='Fitness Medio')
    plt.fill_between(iterations, mean_curve - std_curve, mean_curve + std_curve, 
                     color='blue', alpha=0.2, label='Desviación Estándar')
    plt.yscale('log')
    plt.title(f'Curva de Convergencia Promedio ({len(seeds)} seeds) - Ackley d={dim}')
    plt.xlabel('Iteración')
    plt.ylabel('Mejor Fitness Global (Log)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    os.makedirs('results/viz', exist_ok=True)
    plt.savefig('results/viz/curva_convergencia_promedio.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.boxplot(final_fitnesses, labels=['Vectorizado'], patch_artist=True,
                boxprops=dict(facecolor='#8ab17d', color='black'))
    plt.title('Dispersión del Fitness Final (Boxplot)')
    plt.ylabel('Fitness')
    plt.yscale('log')
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig('results/viz/boxplot_fitness_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráficos del informe generados en 'results/viz/'")

if __name__ == '__main__':
    generate_report_visuals()