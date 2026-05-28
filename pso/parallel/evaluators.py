import numpy as np
import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any
from joblib import Parallel, delayed

class FitnessEvaluator(ABC):
    
    @abstractmethod
    def evaluate(self, particles: np.ndarray, benchmark: Any) -> np.ndarray:
        pass

class SequentialEvaluator(FitnessEvaluator):
    def evaluate(self, particles: np.ndarray, benchmark: Any) -> np.ndarray:
        fitness = [benchmark.evaluate_scalar(p) for p in particles]
        return np.array(fitness)

class ThreadingEvaluator(FitnessEvaluator):
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers

    def evaluate(self, particles: np.ndarray, benchmark: Any) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fitness = list(executor.map(benchmark.evaluate_scalar, particles))
        return np.array(fitness)

class MultiprocessingEvaluator(FitnessEvaluator):
    def __init__(self, max_workers: int = None, chunksize: int = 8):
        self.max_workers = max_workers
        self.chunksize = chunksize

    def evaluate(self, particles: np.ndarray, benchmark: Any) -> np.ndarray:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            fitness = list(executor.map(
                benchmark.evaluate_scalar, 
                particles, 
                chunksize=self.chunksize
            ))
        return np.array(fitness)

class AsyncioEvaluator(FitnessEvaluator):
    def evaluate(self, particles: np.ndarray, benchmark: Any) -> np.ndarray:
        return asyncio.run(self._gather_tasks(particles, benchmark))
        
    async def _gather_tasks(self, particles: np.ndarray, benchmark: Any) -> np.ndarray:
        tasks = [benchmark.evaluate_async(p) for p in particles]
        fitness = await asyncio.gather(*tasks)
        return np.array(fitness)

class VectorizedEvaluator(FitnessEvaluator):
    def evaluate(self, particles: np.ndarray, benchmark: Any) -> np.ndarray:
        return benchmark.evaluate_vectorized(particles)
class JoblibEvaluator(FitnessEvaluator):
    def __init__(self, max_workers: int = -1):
        self.max_workers = max_workers

    def evaluate(self, particles: np.ndarray, benchmark: Any) -> np.ndarray:
        fitness = Parallel(n_jobs=self.max_workers)(
            delayed(benchmark.evaluate_scalar)(p) for p in particles
        )
        return np.array(fitness)