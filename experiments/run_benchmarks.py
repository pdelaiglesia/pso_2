import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pandas as pd
from itertools import product

from pso.core.engine import PSOEngine
from pso.objectives.benchmarks import Sphere, Rosenbrock, Rastrigin, Ackley
from pso.parallel.evaluators import (
    SequentialEvaluator, 
    ThreadingEvaluator, 
    MultiprocessingEvaluator, 
    AsyncioEvaluator, 
    VectorizedEvaluator,
    JoblibEvaluator
)

def run_experiment_suite():
    dimensions_list = [2, 10, 30]
    functions_list = [Sphere, Rosenbrock, Rastrigin, Ackley]
    
    evaluators_dict = {
        'V0_Sequential': SequentialEvaluator(),
        'V1_Threading': ThreadingEvaluator(max_workers=4),
        'V2_Multiprocessing': MultiprocessingEvaluator(max_workers=4, chunksize=8),
        'V3_Asyncio': AsyncioEvaluator(),
        'V4_Vectorized': VectorizedEvaluator(),
        'V5_Joblib': JoblibEvaluator(max_workers=4)
    }
    
    PSO_PARAMS = {
        'num_particles': 50,
        'max_iterations': 100,
        'w': 0.729,
        'c1': 1.49445,
        'c2': 1.49445,
        'boundary_policy': 'clamp',
        'seed': 42 # Fundamental para la reproducibilidad
    }

    dummy_engine = PSOEngine(num_particles=10, dimensions=2, bounds=(-1, 1), evaluator=evaluators_dict['V2_Multiprocessing'])
    dummy_engine.optimize(Sphere(2), max_iterations=2)
    print("Warm-up completado.\n" + "-"*50)

    results = []
    total_combinations = len(functions_list) * len(dimensions_list) * len(evaluators_dict)
    current_run = 1

    for FuncClass, dim in product(functions_list, dimensions_list):
        benchmark = FuncClass(dimensions=dim)
        bounds = benchmark.bounds
        
        print(f"Evaluando: {benchmark.__class__.__name__} | Dimensiones: {dim}")
        
        for eval_name, evaluator in evaluators_dict.items():
            print(f"  [{current_run}/{total_combinations}] Estrategia: {eval_name}...", end=" ", flush=True)
            
            engine = PSOEngine(
                num_particles=PSO_PARAMS['num_particles'],
                dimensions=dim,
                bounds=bounds,
                evaluator=evaluator,
                w=PSO_PARAMS['w'],
                c1=PSO_PARAMS['c1'],
                c2=PSO_PARAMS['c2'],
                boundary_policy=PSO_PARAMS['boundary_policy'],
                seed=PSO_PARAMS['seed']
            )
            
            start_wall = time.perf_counter()
            out = engine.optimize(benchmark, max_iterations=PSO_PARAMS['max_iterations'])
            wall_time = time.perf_counter() - start_wall
            
            avg_eval_time = sum(out['history']['eval_times']) / out['iterations']
            avg_update_time = sum(out['history']['update_times']) / out['iterations']
            
            results.append({
                'Function': benchmark.__class__.__name__,
                'Dimensions': dim,
                'Evaluator': eval_name,
                'Best_Fitness': out['gbest_fitness'],
                'Wall_Time_sec': wall_time,
                'Avg_Eval_Time_sec': avg_eval_time,
                'Avg_Update_Time_sec': avg_update_time,
                'Iterations': out['iterations']
            })
            
            print(f"Done. (Fitness: {out['gbest_fitness']:.2e} | Tiempo: {wall_time:.4f}s)")
            current_run += 1
            
        print("-" * 50)

    os.makedirs('results', exist_ok=True)
    df_results = pd.DataFrame(results)
    
    output_path = 'results/benchmark_results.csv'
    df_results.to_csv(output_path, index=False)
    
    print(f"\nBanco de pruebas finalizado. Resultados guardados en '{output_path}'.")
    
    print("\nResumen rápido (Tiempos totales por evaluador):")
    summary = df_results.groupby('Evaluator')['Wall_Time_sec'].sum().sort_values()
    print(summary)

if __name__ == '__main__':
    run_experiment_suite()