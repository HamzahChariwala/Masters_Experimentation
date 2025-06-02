#!/usr/bin/env python3
import sys
sys.path.insert(0, '../..')
from Optimisation_Formulation.GradBasedTooling.enhanced_optimization import run_enhanced_optimization
import json

# Test 1: Load and validate JSON config
print("Testing JSON config loading...")
with open('test_config.json', 'r') as f:
    config = json.load(f)

print(f"✅ JSON loaded: {config['num_neurons']} neurons, seed {config['seed']}, method {config['neuron_selection']['method']}")

# Test 2: Validate config merge with defaults
try:
    from Optimisation_Formulation.GradBasedTooling.configs.default_config import create_config_from_file
    merged_config = create_config_from_file('test_config.json')
    print(f"✅ Config merge successful: {len(merged_config)} total parameters")
    print(f"   Key params: neurons={merged_config['num_neurons']}, seed={merged_config['seed']}, weights_per_neuron={merged_config['weights_per_neuron']}")
except Exception as e:
    print(f"❌ Config merge failed: {e}")

print("JSON config loading validation complete!") 