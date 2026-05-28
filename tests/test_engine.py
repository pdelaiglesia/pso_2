import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pso.core.engine import PSOEngine
from pso.objectives.benchmarks import Sphere
from pso.parallel.evaluators import VectorizedEvaluator

def test_reproducibility():
    evaluator = VectorizedEvaluator()
    benchmark = Sphere(dimensions=5)
    
    engine1 = PSOEngine(num_particles=20, dimensions=5, bounds=benchmark.bounds, evaluator=evaluator, seed=42)
    res1 = engine1.optimize(benchmark, max_iterations=30)
    
    engine2 = PSOEngine(num_particles=20, dimensions=5, bounds=benchmark.bounds, evaluator=evaluator, seed=42)
    res2 = engine2.optimize(benchmark, max_iterations=30)
    
    assert np.isclose(res1['gbest_fitness'], res2['gbest_fitness']), "Los fitness difieren con la misma semilla."
    assert np.allclose(res1['gbest_position'], res2['gbest_position']), "Las posiciones difieren con la misma semilla."

def test_boundary_handling():
    evaluator = VectorizedEvaluator()
    benchmark = Sphere(dimensions=2) 
    
    engine = PSOEngine(num_particles=50, dimensions=2, bounds=benchmark.bounds, evaluator=evaluator, boundary_policy='clamp', seed=123)
    engine.optimize(benchmark, max_iterations=20)
    
    min_b, max_b = benchmark.bounds
    out_of_bounds = (engine.positions < min_b) | (engine.positions > max_b)
    
    assert not np.any(out_of_bounds), "Box Constraints fallaron: Hay partículas fuera de los límites."

def test_monotonic_evolution():
    evaluator = VectorizedEvaluator()
    benchmark = Sphere(dimensions=10)
    
    engine = PSOEngine(num_particles=30, dimensions=10, bounds=benchmark.bounds, evaluator=evaluator, seed=99)
    res = engine.optimize(benchmark, max_iterations=50)
    
    history_fitness = res['history']['best_fitness']
    
    for i in range(1, len(history_fitness)):
        assert history_fitness[i] <= history_fitness[i-1], "Evolución no monotónica: el fitness global ha empeorado."

def test_correctness_sphere():
    evaluator = VectorizedEvaluator()
    benchmark = Sphere(dimensions=3) 
    
    engine = PSOEngine(num_particles=50, dimensions=3, bounds=benchmark.bounds, evaluator=evaluator, seed=42)
    res = engine.optimize(benchmark, max_iterations=200)
    
    assert res['gbest_fitness'] < 1e-4, f"Fallo de correctitud: No convergió a ~0. Fitness obtenido: {res['gbest_fitness']}"