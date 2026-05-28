import json
import os
import platform
import numpy as np
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

class JSONPersister:
    @staticmethod
    def save_experiment(experiment_name: str, config: dict, results: dict, output_dir: str = 'results'):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        payload = {
            "metadata": {
                "experiment": experiment_name,
                "timestamp": timestamp,
                "system_info": {
                    "os": platform.system(),
                    "processor": platform.processor(),
                    "python_version": platform.python_version()
                }
            },
            "configuration": config,
            "results": results
        }
        
        filepath = os.path.join(output_dir, f"{experiment_name}_{timestamp}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, cls=NumpyEncoder)
            
        return filepath