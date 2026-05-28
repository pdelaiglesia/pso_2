import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pso.viz.plotting import plot_speedup_bar

def analyze_benchmarks(csv_path: str = 'results/benchmark_results.csv'):
    if not os.path.exists(csv_path):
        print(f"No se ha encontrado el archivo {csv_path}.")
        print("Asegúrate de ejecutar 'python experiments/run_benchmarks.py' primero.")
        return

    print("Cargando resultados de los benchmarks...\n")
    df = pd.read_csv(csv_path)

    print("TIEMPOS TOTALES POR ESTRATEGIA Y FUNCIÓN")
    time_pivot = df.pivot_table(
        index=['Function', 'Dimensions'], 
        columns='Evaluator', 
        values='Wall_Time_sec', 
        aggfunc='mean'
    )
    print(time_pivot.round(4).to_string())
    print("\n" + "-"*60 + "\n")

    print("SPEEDUP Y EFICIENCIA VS BASELINE (V0_Sequential)")
    
    if 'V0_Sequential' in time_pivot.columns:
        speedup_df = pd.DataFrame(index=time_pivot.index)
        efficiency_df = pd.DataFrame(index=time_pivot.index)
        
        NUM_CORES = 4 
        
        for col in time_pivot.columns:
            speedup_df[f'Speedup_{col}'] = time_pivot['V0_Sequential'] / time_pivot[col]
            efficiency_df[f'Efic_{col}'] = speedup_df[f'Speedup_{col}'] / NUM_CORES
            
        print("SPEEDUP")
        print(speedup_df.round(2).to_string())
        
        print("\nEFICIENCIA")
        cols_paralelas = [c for c in efficiency_df.columns if any(x in c for x in ['V1', 'V2', 'V5'])]
        print(efficiency_df[cols_paralelas].round(3).to_string())
    else:
        print("No se encontró 'V0_Sequential' para calcular el speedup base")
    
    print("\n" + "-"*60 + "\n")

    print("GENERANDO GRÁFICO DE RENDIMIENTO GLOBAL")
    total_times = df.groupby('Evaluator')['Wall_Time_sec'].sum().to_dict()
    
    out_dir = "results/viz"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "global_speedup_comparison.png")
    
    plot_speedup_bar(times=total_times, title="Comparativa Global de Estrategias Concurrentes", filepath=out_file)
    
    print("\nDESGLOSE DE OVERHEAD MEDIO POR ESTRATEGIA")
    overhead_df = df.groupby('Evaluator')[['Avg_Eval_Time_sec', 'Avg_Update_Time_sec']].mean()
    print(overhead_df.round(6).to_string())
    print("\nAnálisis completado. Revisa la carpeta 'results/viz/' para ver el gráfico.")

if __name__ == '__main__':
    analyze_benchmarks()