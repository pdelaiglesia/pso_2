import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from pso.core.engine import PSOEngine
from pso.objectives.benchmarks import Sphere, Rosenbrock, Rastrigin, Ackley
from pso.parallel.evaluators import VectorizedEvaluator

class TrackingPSOEngine(PSOEngine):
    """
    Extiende el motor base para capturar el histórico de posiciones
    en cada iteración. Uso exclusivo para visualización (penaliza memoria).
    """
    def optimize(self, benchmark, max_iterations: int, tolerance: float = 1e-6):
        self.history['positions'] = []
        
        for _ in range(max_iterations):
            self.history['positions'].append(np.copy(self.positions))
            
            current_fitness = self.evaluator.evaluate(self.positions, benchmark)
            
            improved = current_fitness < self.pbest_fitness
            self.pbest_fitness[improved] = current_fitness[improved]
            self.pbest_positions[improved] = self.positions[improved]
            
            best_idx = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[best_idx] < self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitness[best_idx]
                self.gbest_position = np.copy(self.pbest_positions[best_idx])
                
            r1 = np.random.rand(self.num_particles, self.dimensions)
            r2 = np.random.rand(self.num_particles, self.dimensions)
            
            self.velocities = (self.w * self.velocities + 
                               self.c1 * r1 * (self.pbest_positions - self.positions) + 
                               self.c2 * r2 * (self.gbest_position - self.positions))
            self.positions += self.velocities
            
            self.positions, self.velocities = self.apply_bounds(self.positions, self.velocities, self.bounds)
            
            if self.gbest_fitness <= tolerance:
                self.history['positions'].append(np.copy(self.positions))
                break
                
        return self.history

def create_pso_animation(benchmark_class, num_particles=30, max_iterations=50):

    dim = 2
    benchmark = benchmark_class(dimensions=dim)
    bounds = benchmark.bounds
    
    evaluator = VectorizedEvaluator()
    
    engine = TrackingPSOEngine(
        num_particles=num_particles,
        dimensions=dim,
        bounds=bounds,
        evaluator=evaluator,
        w=0.6, c1=1.5, c2=1.5,
        boundary_policy='reflect',
        seed=123
    )
    
    print(f"Ejecutando PSO sobre {benchmark.__class__.__name__}...")
    history = engine.optimize(benchmark, max_iterations=max_iterations)
    positions_history = history['positions']
    
    print("Calculando superficie de la función...")
    x = np.linspace(bounds[0], bounds[1], 150)
    y = np.linspace(bounds[0], bounds[1], 150)
    X, Y = np.meshgrid(x, y)
    
    grid_points = np.c_[X.ravel(), Y.ravel()]
    Z = evaluator.evaluate(grid_points, benchmark).reshape(X.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    contour = ax.contourf(X, Y, Z, levels=50, cmap='magma', alpha=0.9)
    fig.colorbar(contour, ax=ax, label='Fitness (Coste)')
    
    scatter = ax.scatter([], [], c='cyan', edgecolors='white', s=40, label='Enjambre')
    gbest_marker = ax.scatter([], [], c='gold', marker='*', edgecolors='black', s=250, label='Mejor Global', zorder=5)
    
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_xlabel('Dimensión 1')
    ax.set_ylabel('Dimensión 2')
    ax.legend(loc='upper right')
    title = ax.set_title('')
    
    def update(frame):
        current_positions = positions_history[frame]
        scatter.set_offsets(current_positions)
        
        current_fitness = evaluator.evaluate(current_positions, benchmark)
        best_idx = np.argmin(current_fitness)
        gbest_marker.set_offsets(current_positions[best_idx])
        
        title.set_text(f'{benchmark.__class__.__name__} - Iteración {frame + 1}/{len(positions_history)}')
        return scatter, gbest_marker, title

    print("Renderizando animación")
    anim = FuncAnimation(fig, update, frames=len(positions_history), interval=100, blit=True)
    
    os.makedirs('results/viz', exist_ok=True)
    output_path = f'results/viz/{benchmark.__class__.__name__.lower()}_pso.gif'
    
    anim.save(output_path, writer=PillowWriter(fps=10))
    print(f"Animación guardada en: {output_path}")
    plt.close()
def create_pso_3d_surface_animation(benchmark_class, num_particles=30, max_iterations=50):
  
    dim = 2
    benchmark = benchmark_class(dimensions=dim)
    bounds = benchmark.bounds
    evaluator = VectorizedEvaluator()
    
    engine = TrackingPSOEngine(
        num_particles=num_particles, dimensions=dim, bounds=bounds,
        evaluator=evaluator, w=0.6, c1=1.5, c2=1.5, boundary_policy='reflect', seed=123
    )
    
    print(f"Ejecutando PSO (Superficie 3D) sobre {benchmark.__class__.__name__}...")
    history = engine.optimize(benchmark, max_iterations=max_iterations)
    positions_history = history['positions']
    
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    grid_points = np.c_[X.ravel(), Y.ravel()]
    Z = evaluator.evaluate(grid_points, benchmark).reshape(X.shape)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='magma', alpha=0.5, edgecolor='none')
    fig.colorbar(surf, ax=ax, label='Fitness (Coste)', shrink=0.5, aspect=10)
    
    scatter = ax.scatter([], [], [], c='cyan', edgecolors='black', s=50, depthshade=True, label='Enjambre')
    gbest_marker = ax.scatter([], [], [], c='gold', marker='*', edgecolors='black', s=300, label='Mejor Global', zorder=10)
    
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_zlim(np.min(Z), np.max(Z)) 
    
    ax.set_xlabel('Dimensión 1')
    ax.set_ylabel('Dimensión 2')
    ax.set_zlabel('Fitness')
    ax.legend()
    title = ax.set_title('')
    
    def update(frame):
        current_positions = positions_history[frame]
        current_fitness = evaluator.evaluate(current_positions, benchmark)
        
        scatter._offsets3d = (current_positions[:, 0], current_positions[:, 1], current_fitness)
        
        best_idx = np.argmin(current_fitness)
        gbest_marker._offsets3d = ([current_positions[best_idx, 0]], 
                                   [current_positions[best_idx, 1]], 
                                   [current_fitness[best_idx]])
        
        title.set_text(f'{benchmark.__class__.__name__} (d=2) Superficie 3D - Iteración {frame + 1}')
        return scatter, gbest_marker, title

    print("Renderizando animación 3D")
    anim = FuncAnimation(fig, update, frames=len(positions_history), interval=100, blit=False)
    
    output_path = f'results/viz/{benchmark.__class__.__name__.lower()}_pso_3d_surface.gif'
    anim.save(output_path, writer=PillowWriter(fps=10))
    print(f"Animación 3D guardada en: {output_path}")
    plt.close()


def create_pso_3d_scatter_animation(benchmark_class, num_particles=50, max_iterations=50):
    dim = 3
    benchmark = benchmark_class(dimensions=dim)
    bounds = benchmark.bounds
    evaluator = VectorizedEvaluator()
    
    engine = TrackingPSOEngine(
        num_particles=num_particles, dimensions=dim, bounds=bounds,
        evaluator=evaluator, w=0.6, c1=1.5, c2=1.5, boundary_policy='reflect', seed=42
    )
    
    print(f"Ejecutando PSO (Espacio 3D puro) sobre {benchmark.__class__.__name__}...")
    history = engine.optimize(benchmark, max_iterations=max_iterations)
    positions_history = history['positions']
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter([], [], [], c='royalblue', s=40, label='Enjambre')
    gbest_marker = ax.scatter([], [], [], c='red', marker='X', s=150, label='Mejor Global')
    
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_zlim(bounds[0], bounds[1])
    
    ax.set_xlabel('Dimensión 1 (X)')
    ax.set_ylabel('Dimensión 2 (Y)')
    ax.set_zlabel('Dimensión 3 (Z)')
    ax.legend()
    title = ax.set_title('')
    
    def update(frame):
        current_positions = positions_history[frame]
        current_fitness = evaluator.evaluate(current_positions, benchmark)
        
        scatter._offsets3d = (current_positions[:, 0], current_positions[:, 1], current_positions[:, 2])
        
        best_idx = np.argmin(current_fitness)
        gbest_marker._offsets3d = ([current_positions[best_idx, 0]], 
                                   [current_positions[best_idx, 1]], 
                                   [current_positions[best_idx, 2]])
        
        title.set_text(f'{benchmark.__class__.__name__} (d=3) Nube 3D - Iteración {frame + 1}')
        return scatter, gbest_marker, title

    anim = FuncAnimation(fig, update, frames=len(positions_history), interval=100, blit=False)
    output_path = f'results/viz/{benchmark.__class__.__name__.lower()}_pso_3d_scatter.gif'
    anim.save(output_path, writer=PillowWriter(fps=10))
    print(f"Animación d=3 guardada en: {output_path}")
    plt.close()

if __name__ == '__main__':
    create_pso_animation(Sphere)
    create_pso_animation(Rastrigin)
    
    create_pso_3d_surface_animation(Sphere)
    create_pso_3d_surface_animation(Rosenbrock)
    
    create_pso_3d_scatter_animation(Ackley)