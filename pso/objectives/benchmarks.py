import numpy as np
import asyncio
from abc import ABC, abstractmethod

class BenchmarkFunction(ABC):
    
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        self.bounds = self._get_bounds()
        self.global_optimum = 0.0 

    @abstractmethod
    def _get_bounds(self) -> tuple[float, float]:
        pass

    @abstractmethod
    def evaluate_scalar(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def evaluate_vectorized(self, x: np.ndarray) -> np.ndarray:
        pass

    async def evaluate_async(self, x: np.ndarray, latency: float = 0.01) -> float:
        """
        Evaluación asíncrona para la variante V3 (Asyncio).
        Simula una carga de I/O o latencia de red.
        """
        await asyncio.sleep(latency)
        return self.evaluate_scalar(x)

class Sphere(BenchmarkFunction):
    def _get_bounds(self) -> tuple[float, float]:
        return (-5.12, 5.12)

    def evaluate_scalar(self, x: np.ndarray) -> float:
        return np.sum(x**2)

    def evaluate_vectorized(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x**2, axis=1)

class Rastrigin(BenchmarkFunction):
    def _get_bounds(self) -> tuple[float, float]:
        return (-5.12, 5.12)

    def evaluate_scalar(self, x: np.ndarray) -> float:
        d = self.dimensions
        return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))

    def evaluate_vectorized(self, x: np.ndarray) -> np.ndarray:
        d = self.dimensions
        return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x), axis=1)

class Rosenbrock(BenchmarkFunction):
    def _get_bounds(self) -> tuple[float, float]:
        return (-5.0, 10.0) 

    def evaluate_scalar(self, x: np.ndarray) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def evaluate_vectorized(self, x: np.ndarray) -> np.ndarray:
        term1 = 100.0 * (x[:, 1:] - x[:, :-1]**2)**2
        term2 = (1 - x[:, :-1])**2
        return np.sum(term1 + term2, axis=1)

class Ackley(BenchmarkFunction):
    def _get_bounds(self) -> tuple[float, float]:
        return (-32.768, 32.768)

    def evaluate_scalar(self, x: np.ndarray) -> float:
        d = self.dimensions
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        
        return term1 + term2 + 20.0 + np.e

    def evaluate_vectorized(self, x: np.ndarray) -> np.ndarray:
        d = self.dimensions
        sum_sq = np.sum(x**2, axis=1)
        sum_cos = np.sum(np.cos(2 * np.pi * x), axis=1)
        
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        
        return term1 + term2 + 20.0 + np.e