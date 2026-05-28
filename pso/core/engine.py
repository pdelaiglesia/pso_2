import time
import numpy as np
from typing import Any, Dict

class BoundaryPolicy:
    
    @staticmethod
    def clamp(positions: np.ndarray, velocities: np.ndarray, bounds: tuple[float, float]):
        """Corta la posición en el límite y anula la velocidad en ese eje."""
        min_b, max_b = bounds
        out_min = positions < min_b
        out_max = positions > max_b
        velocities[out_min | out_max] = 0.0 
        np.clip(positions, min_b, max_b, out=positions)
        return positions, velocities

    @staticmethod
    def reflect(positions: np.ndarray, velocities: np.ndarray, bounds: tuple[float, float]):
        min_b, max_b = bounds
        out_min = positions < min_b
        out_max = positions > max_b
        positions[out_min] = 2 * min_b - positions[out_min]
        velocities[out_min] *= -1.0
        positions[out_max] = 2 * max_b - positions[out_max]
        velocities[out_max] *= -1.0
        return positions, velocities

class PSOEngine:
    
    def __init__(self, 
                 num_particles: int, 
                 dimensions: int, 
                 bounds: tuple[float, float], 
                 evaluator: Any,
                 w: float = 0.5, 
                 c1: float = 1.5, 
                 c2: float = 1.5,
                 boundary_policy: str = 'clamp',
                 seed: int = None):
        
        if seed is not None:
            np.random.seed(seed)
            
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.evaluator = evaluator
        
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.apply_bounds = BoundaryPolicy.clamp if boundary_policy == 'clamp' else BoundaryPolicy.reflect
        
        self.positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
        v_range = abs(bounds[1] - bounds[0]) * 0.1
        self.velocities = np.random.uniform(-v_range, v_range, (num_particles, dimensions))
        
        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitness = np.full(num_particles, np.inf)
        
        self.gbest_position = np.zeros(dimensions)
        self.gbest_fitness = np.inf
        
        self.history = {
            'best_fitness': [],
            'eval_times': [],
            'update_times': []
        }

    def optimize(self, benchmark: Any, max_iterations: int, tolerance: float = 1e-6, max_stagnation: int = 15) -> Dict[str, Any]:
        
        start_total = time.perf_counter()
        
        stagnation_counter = 0
        last_gbest_fitness = np.inf
        
        for iteration in range(max_iterations):
            t0 = time.perf_counter()
            current_fitness = self.evaluator.evaluate(self.positions, benchmark)
            eval_time = time.perf_counter() - t0
            
            t1 = time.perf_counter()
            
            improved_mask = current_fitness < self.pbest_fitness
            self.pbest_fitness[improved_mask] = current_fitness[improved_mask]
            self.pbest_positions[improved_mask] = self.positions[improved_mask]
            
            best_current_idx = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[best_current_idx] < self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitness[best_current_idx]
                self.gbest_position = np.copy(self.pbest_positions[best_current_idx])
            
            r1 = np.random.rand(self.num_particles, self.dimensions)
            r2 = np.random.rand(self.num_particles, self.dimensions)
            
            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            
            self.velocities = self.w * self.velocities + cognitive + social
            self.positions += self.velocities
            
            self.positions, self.velocities = self.apply_bounds(self.positions, self.velocities, self.bounds)
            
            update_time = time.perf_counter() - t1
            
            self.history['best_fitness'].append(self.gbest_fitness)
            self.history['eval_times'].append(eval_time)
            self.history['update_times'].append(update_time)
            
            if self.gbest_fitness <= tolerance:
                stop_reason = 'tolerance'
                break
                
            if abs(last_gbest_fitness - self.gbest_fitness) < 1e-8:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                last_gbest_fitness = self.gbest_fitness
                
            if stagnation_counter >= max_stagnation:
                stop_reason = 'stagnation'
                break
                
        else:
            stop_reason = 'max_iterations'
                
        total_time = time.perf_counter() - start_total
        
        return {
            'gbest_position': self.gbest_position,
            'gbest_fitness': self.gbest_fitness,
            'iterations': iteration + 1,
            'stop_reason': stop_reason,
            'history': self.history,
            'total_time': total_time
        }