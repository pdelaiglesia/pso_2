import numpy as np
from abc import ABC, abstractmethod

class Topology(ABC):
    
    @abstractmethod
    def get_social_targets(self, pbest_positions: np.ndarray, pbest_fitness: np.ndarray) -> np.ndarray:
        pass

class GlobalBest(Topology):
    
    def get_social_targets(self, pbest_positions: np.ndarray, pbest_fitness: np.ndarray) -> np.ndarray:
        best_idx = np.argmin(pbest_fitness)
        gbest_position = pbest_positions[best_idx]
        
        num_particles = pbest_positions.shape[0]
        return np.tile(gbest_position, (num_particles, 1))

class LocalBestRing(Topology):
    
    def get_social_targets(self, pbest_positions: np.ndarray, pbest_fitness: np.ndarray) -> np.ndarray:
        num_particles = pbest_positions.shape[0]
        social_targets = np.zeros_like(pbest_positions)
        
        for i in range(num_particles):
            left_neighbor = (i - 1) % num_particles
            right_neighbor = (i + 1) % num_particles
            
            neighborhood_indices = [left_neighbor, i, right_neighbor]
            best_local_idx = neighborhood_indices[np.argmin(pbest_fitness[neighborhood_indices])]
            
            social_targets[i] = pbest_positions[best_local_idx]
            
        return social_targets