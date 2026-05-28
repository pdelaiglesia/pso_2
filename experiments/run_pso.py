import sys
import os
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pso.core.engine import PSOEngine
from pso.objectives.benchmarks import Ackley, Sphere, Rosenbrock, Rastrigin
from pso.parallel.evaluators import VectorizedEvaluator, SequentialEvaluator
from pso.viz.plotting import plot_convergence

BENCHMARKS = {
    'sphere': Sphere,
    'rosenbrock': Rosenbrock,
    'rastrigin': Rastrigin,
    'ackley': Ackley
}

EVALUATORS = {
    'vectorized': VectorizedEvaluator,
    'sequential': SequentialEvaluator
}

def setup_logging(log_level: str):
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="CLI para ejecutar PSO de forma individual.")
    parser.add_argument('--benchmark', type=str, default='ackley', choices=BENCHMARKS.keys(), help="Función a optimizar")
    parser.add_argument('--dim', type=int, default=30, help="Dimensiones del problema")
    parser.add_argument('--particles', type=int, default=50, help="Número de partículas en el enjambre")
    parser.add_argument('--iterations', type=int, default=200, help="Iteraciones máximas")
    parser.add_argument('--evaluator', type=str, default='vectorized', choices=EVALUATORS.keys(), help="Estrategia de evaluación")
    parser.add_argument('--seed', type=int, default=1024, help="Semilla para reproducibilidad")
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING'], help="Nivel de verbosidad")

    args = parser.parse_args()
    setup_logging(args.log_level)

    logging.info("Iniciando Ejecución de PSO")
    logging.info(f"Configuración: Función={args.benchmark.upper()}, Dim={args.dim}, "
                 f"Partículas={args.particles}, Iteraciones={args.iterations}, "
                 f"Evaluador={args.evaluator}, Seed={args.seed}")

    benchmark_class = BENCHMARKS[args.benchmark]
    benchmark = benchmark_class(dimensions=args.dim)
    evaluator = EVALUATORS[args.evaluator]()

    engine = PSOEngine(
        num_particles=args.particles,
        dimensions=args.dim,
        bounds=benchmark.bounds,
        evaluator=evaluator,
        w=0.729, c1=1.49445, c2=1.49445,
        boundary_policy='clamp',
        seed=args.seed
    )

    logging.info("Optimizando")
    resultados = engine.optimize(benchmark, max_iterations=args.iterations, tolerance=1e-8)

    logging.info("Resultados de la Optimización:")
    logging.info(f"Criterio de parada   : {resultados.get('stop_reason', 'max_iterations')}")
    logging.info(f"Iteraciones consumidas: {resultados['iterations']}")
    logging.info(f"Tiempo total         : {resultados['total_time']:.4f} s")
    logging.info(f"Mejor Fitness Final  : {resultados['gbest_fitness']:.4e}")

    out_dir = "results/viz"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"convergencia_{args.benchmark}_d{args.dim}.png")
    
    plot_convergence(history=resultados['history'], title=f"Convergencia PSO en {args.benchmark.capitalize()} (d={args.dim})", filepath=out_file)
    logging.info(f"Ejecución terminada correctamente. Gráfico guardado en: {out_file}")

if __name__ == '__main__':
    main()