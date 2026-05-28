import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from itertools import product
from tqdm import tqdm

from pso.core.engine import PSOEngine
from pso.objectives.benchmarks import Rosenbrock
from pso.parallel.evaluators import VectorizedEvaluator
from pso.io.persistence import JSONPersister

def run_grid_search():
    print("Iniciando Grid Search de Hiperparámetros...")
    
    grid = {
        'w': [0.5, 0.729, 0.9],
        'c1': [1.0, 1.49445, 2.0],
        'c2': [1.0, 1.49445, 2.0],
        'num_particles': [30, 50]
    }
    
    seeds = [42, 100, 2026, 9999, 12345] 
    
    dim = 10
    max_iterations = 100
    benchmark = Rosenbrock(dimensions=dim)
    evaluator = VectorizedEvaluator()
    
    keys = list(grid.keys())
    combinations = list(product(*[grid[k] for k in keys]))
    
    print(f"Total de configuraciones a evaluar: {len(combinations)}")
    print(f"Ejecuciones por configuración (seeds): {len(seeds)}")
    print(f"Total de ejecuciones PSO: {len(combinations) * len(seeds)}\n")
    
    best_overall_fitness = np.inf
    best_config = None
    all_results = []
    
    for combo in tqdm(combinations, desc="Evaluando Grid"):
        params = dict(zip(keys, combo))
        
        fitness_across_seeds = []
        iterations_across_seeds = []
        times_across_seeds = []
        
        for current_seed in seeds:
            engine = PSOEngine(
                num_particles=params['num_particles'],
                dimensions=dim,
                bounds=benchmark.bounds,
                evaluator=evaluator,
                w=params['w'],
                c1=params['c1'],
                c2=params['c2'],
                boundary_policy='clamp',
                seed=current_seed
            )
            
            out = engine.optimize(benchmark, max_iterations=max_iterations)
            
            fitness_across_seeds.append(out['gbest_fitness'])
            iterations_across_seeds.append(out['iterations'])
            times_across_seeds.append(out['total_time'])
            
        mean_fitness = np.mean(fitness_across_seeds)
        std_fitness = np.std(fitness_across_seeds)
        mean_iterations = np.mean(iterations_across_seeds)
        std_iterations = np.std(iterations_across_seeds)
        mean_time = np.mean(times_across_seeds)
        std_time = np.std(times_across_seeds)
        
        config_result = {
            "parameters": params,
            "metrics": {
                "fitness": {
                    "mean": mean_fitness,
                    "std": std_fitness,
                    "raw_per_seed": fitness_across_seeds
                },
                "iterations": {
                    "mean": mean_iterations,
                    "std": std_iterations,
                    "raw_per_seed": iterations_across_seeds
                },
                "total_time_sec": {
                    "mean": mean_time,
                    "std": std_time,
                    "raw_per_seed": times_across_seeds
                }
            }
        }
        all_results.append(config_result)
        
        if mean_fitness < best_overall_fitness:
            best_overall_fitness = mean_fitness
            best_config = params
            
    print("\n" + "="*50)
    print("Grid Search Finalizado.")
    print("Mejor configuración encontrada:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(f"Fitness medio esperado: {best_overall_fitness:.4e}")
    print("="*50)
    
    experiment_config = {
        "function": benchmark.__class__.__name__,
        "dimensions": dim,
        "max_iterations": max_iterations,
        "seeds_used": seeds,
        "grid_space": grid
    }
    
    experiment_results = {
        "best_configuration": best_config,
        "best_mean_fitness": best_overall_fitness,
        "all_combinations": all_results
    }
    
    saved_path = JSONPersister.save_experiment(
        experiment_name="grid_search_rosenbrock",
        config=experiment_config,
        results=experiment_results
    )
    
    print(f"\nTodos los resultados y metadatos guardados en: {saved_path}")

if __name__ == '__main__':
    run_grid_search()