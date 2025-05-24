"""Configuration settings for hyperparameter tuning."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import yaml

@dataclass
class TuningConfig:
    # Number of agents to train per parameter set
    agents_per_trial: int = 5
    
    # Base directory for storing hyperparameter tuning results
    base_output_dir: str = "Agent_Storage/Hyperparameters"
    
    # Optuna study settings
    study_name: str = "dqn_hyperparameter_optimization"
    storage_name: Optional[str] = None  # Set to "sqlite:///study.db" to persist
    n_trials: int = 100
    
    # Training settings
    reduced_timesteps_factor: float = 1.0  # Factor to reduce timesteps by
    
    # Parameter search space from optuna config
    bayesian_optimization: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TuningConfig':
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @property
    def param_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Convert bayesian_optimization config to param_ranges format."""
        ranges = {}
        for param_name, config in self.bayesian_optimization.items():
            ranges[param_name] = {
                "type": "int" if config["distribution"].startswith("int") else "float",
                "low": config["min"],
                "high": config["max"],
                "log": config["distribution"] == "loguniform"
            }
        return ranges

# Default parameter ranges for optimization
DEFAULT_PARAM_RANGES = {
    "model.buffer_size": {
        "type": "int",
        "low": 50000,
        "high": 1000000,
        "log": True
    },
    "model.learning_starts": {
        "type": "int",
        "low": 10000,
        "high": 200000,
        "log": True
    },
    "model.batch_size": {
        "type": "int",
        "low": 32,
        "high": 512,
        "log": True
    },
    "model.learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-3,
        "log": True
    },
    "model.gamma": {
        "type": "float",
        "low": 0.5,
        "high": 0.999,
        "log": False
    },
    "model.train_freq": {
        "type": "int",
        "low": 1,
        "high": 10,
        "log": False
    },
    "model.target_update_interval": {
        "type": "int",
        "low": 500,
        "high": 10000,
        "log": True
    },
    "model.exploration_fraction": {
        "type": "float",
        "low": 0.1,
        "high": 0.9,
        "log": False
    },
    "model.exploration_final_eps": {
        "type": "float",
        "low": 0.01,
        "high": 0.2,
        "log": True
    }
} 