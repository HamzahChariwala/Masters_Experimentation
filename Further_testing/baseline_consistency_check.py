import json
import numpy as np
from pathlib import Path

def check_baseline_consistency():
    """
    Check if noising and denoising experiments have consistent baselines 
    for the same environment/state combinations.
    """
    
    # Test the environment we've been analyzing
    env_name = 'MiniGrid_LavaCrossingS11N5_v0_81102_7_8_0_0106'
    
    base_path = Path('Agent_Storage/LavaTests/NoDeath/0.100_penalty/0.100_penalty-v6/circuit_verification/results/confidence_margin_magnitude')
    noising_file = base_path / 'noising' / f'{env_name}.json'
    denoising_file = base_path / 'denoising' / f'{env_name}.json'
    
    print(f"=== BASELINE CONSISTENCY CHECK FOR {env_name} ===\n")
    
    with open(noising_file) as f:
        noising_data = json.load(f)
    with open(denoising_file) as f:
        denoising_data = json.load(f)
    
    # Get first experiment from each
    noising_first_exp = list(noising_data.keys())[0]
    denoising_first_exp = list(denoising_data.keys())[0]
    
    noising_baseline = noising_data[noising_first_exp]['results']['baseline_output'][0]
    denoising_baseline = denoising_data[denoising_first_exp]['results']['baseline_output'][0]
    
    print(f"Noising baseline:   {noising_baseline}")
    print(f"Denoising baseline: {denoising_baseline}")
    
    # Check if they are the same
    are_equal = np.allclose(noising_baseline, denoising_baseline, atol=1e-10)
    print(f"\nBaselines are identical: {are_equal}")
    
    if not are_equal:
        diff = np.array(denoising_baseline) - np.array(noising_baseline)
        print(f"Difference (denoising - noising): {diff}")
        print(f"Max absolute difference: {np.max(np.abs(diff))}")
    
    print("\n" + "="*80)
    
    # Check multiple experiments within each type
    print("\n=== CONSISTENCY WITHIN NOISING EXPERIMENTS ===")
    noising_baselines = []
    for i, exp_key in enumerate(list(noising_data.keys())[:5]):
        baseline = noising_data[exp_key]['results']['baseline_output'][0]
        noising_baselines.append(baseline)
        print(f"Exp {i+1}: {baseline}")
    
    # Check if all noising baselines are the same
    noising_consistent = all(np.allclose(b, noising_baselines[0], atol=1e-10) for b in noising_baselines)
    print(f"\nAll noising baselines consistent: {noising_consistent}")
    
    print("\n=== CONSISTENCY WITHIN DENOISING EXPERIMENTS ===")
    denoising_baselines = []
    for i, exp_key in enumerate(list(denoising_data.keys())[:5]):
        baseline = denoising_data[exp_key]['results']['baseline_output'][0]
        denoising_baselines.append(baseline)
        print(f"Exp {i+1}: {baseline}")
    
    # Check if all denoising baselines are the same
    denoising_consistent = all(np.allclose(b, denoising_baselines[0], atol=1e-10) for b in denoising_baselines)
    print(f"\nAll denoising baselines consistent: {denoising_consistent}")
    
    print("\n" + "="*80)
    
    # Now check a few other environments
    print("\n=== CHECKING OTHER ENVIRONMENTS ===")
    
    other_envs = [
        'MiniGrid_LavaCrossingS11N5_v0_81109_2_3_1_0115',
        'MiniGrid_LavaCrossingS11N5_v0_81103_4_1_0_0026'
    ]
    
    for env in other_envs:
        print(f"\n--- Environment: {env} ---")
        
        noising_file = base_path / 'noising' / f'{env}.json'
        denoising_file = base_path / 'denoising' / f'{env}.json'
        
        try:
            with open(noising_file) as f:
                n_data = json.load(f)
            with open(denoising_file) as f:
                d_data = json.load(f)
            
            n_baseline = n_data[list(n_data.keys())[0]]['results']['baseline_output'][0]
            d_baseline = d_data[list(d_data.keys())[0]]['results']['baseline_output'][0]
            
            print(f"Noising:   {n_baseline}")
            print(f"Denoising: {d_baseline}")
            
            are_equal = np.allclose(n_baseline, d_baseline, atol=1e-10)
            print(f"Baselines match: {are_equal}")
            
            if not are_equal:
                diff = np.array(d_baseline) - np.array(n_baseline)
                print(f"Max difference: {np.max(np.abs(diff))}")
                
        except FileNotFoundError as e:
            print(f"File not found: {e}")
    
    return {
        'main_env_consistent': are_equal,
        'noising_internally_consistent': noising_consistent,
        'denoising_internally_consistent': denoising_consistent
    }

if __name__ == "__main__":
    results = check_baseline_consistency()
    
    print(f"\n\n=== SUMMARY ===")
    print(f"Noising vs Denoising baselines match: {results['main_env_consistent']}")
    print(f"Noising experiments internally consistent: {results['noising_internally_consistent']}")
    print(f"Denoising experiments internally consistent: {results['denoising_internally_consistent']}")
    
    if not results['main_env_consistent']:
        print("\n🚨 CRITICAL BUG CONFIRMED: Noising and denoising experiments use different baseline states!")
        print("This explains the inverse relationship - they're comparing different things!") 