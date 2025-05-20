import os
import sys
import json
import argparse

# Add the root directory to sys.path to ensure proper imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_integration_test(agent_path, env_id="MiniGrid-LavaCrossingS11N5-v0", seed=81102):
    """
    Run an end-to-end integration test of the evaluation system.
    
    Args:
        agent_path (str): Path to the agent directory for evaluation
        env_id (str): Environment ID to use
        seed (int): Random seed for reproducibility
    """
    print(f"Running integration test with agent: {agent_path}")
    print(f"Environment: {env_id}, Seed: {seed}\n")
    
    # Import the main evaluation function
    from Agent_Evaluation.generate_evaluations import single_env_evals
    
    # Run the evaluation with our fixed parameters
    # We'll set generate_plot=True for better visualization
    # debug=True for detailed output and force_dijkstra=True to ensure it runs
    single_env_evals(
        agent_path=agent_path,
        env_id=env_id,
        seed=seed,
        generate_plot=True,
        debug=True,
        force_dijkstra=True
    )
    
    # Check the output files
    print("\nVerifying output files...")
    
    # Check for Dijkstra output
    dijkstra_file = os.path.join(
        project_root, "Behaviour_Specification", "Evaluations", f"{env_id}-{seed}.json"
    )
    
    if os.path.exists(dijkstra_file):
        print(f"✓ Dijkstra output file exists: {dijkstra_file}")
        
        # Check that the file contains the performance metrics at the top
        with open(dijkstra_file, 'r') as f:
            data = json.load(f)
            if "performance" in data and data.keys().__iter__().__next__() == "performance":
                print(f"✓ Dijkstra file has performance metrics at the top")
            else:
                print(f"✗ Dijkstra file does not have performance metrics at the top")
    else:
        print(f"✗ Dijkstra output file not found: {dijkstra_file}")
    
    # Check for summary files - these should NOT exist
    summary_files = []
    for mode in ["standard", "conservative", "dangerous_1", "dangerous_2", "dangerous_3", "dangerous_4", "dangerous_5"]:
        summary_file = os.path.join(
            project_root, "Behaviour_Specification", "Evaluations", f"{env_id}-{seed}_{mode}.json"
        )
        if os.path.exists(summary_file):
            summary_files.append(summary_file)
    
    if summary_files:
        print(f"✗ Found {len(summary_files)} separate summary files (these should not exist):")
        for file in summary_files:
            print(f"  - {file}")
    else:
        print(f"✓ No separate summary files found (as expected)")
    
    # Check agent evaluation output
    agent_base_name = os.path.basename(agent_path.rstrip('/'))
    agent_file = os.path.join(
        project_root, "Agent_Evaluation", "Results", f"{agent_base_name}-{env_id}-{seed}.json"
    )
    
    if os.path.exists(agent_file):
        print(f"✓ Agent evaluation file exists: {agent_file}")
        
        # Analyze the file to check for correct risky_diagonal values
        with open(agent_file, 'r') as f:
            data = json.load(f)
            
            # Count states with risky diagonals
            risky_count = 0
            non_risky_count = 0
            diagonal_action_count = 0
            
            for state_key, state_data in data.items():
                if "next_step" in state_data:
                    next_step = state_data["next_step"]
                    if next_step["action"] in [3, 4]:  # Diagonal actions
                        diagonal_action_count += 1
                        if next_step["risky_diagonal"]:
                            risky_count += 1
                        else:
                            non_risky_count += 1
            
            print(f"✓ Found {diagonal_action_count} diagonal actions in the results")
            print(f"  - {risky_count} identified as risky")
            print(f"  - {non_risky_count} identified as safe")
    else:
        print(f"✗ Agent evaluation file not found: {agent_file}")
    
    print("\nIntegration test completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an integration test of the evaluation system")
    parser.add_argument("--path", type=str, required=True, 
                        help="Path to the agent directory for evaluation")
    parser.add_argument("--env", type=str, default="MiniGrid-LavaCrossingS11N5-v0",
                        help="Environment ID to use")
    parser.add_argument("--seed", type=int, default=81102,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    run_integration_test(args.path, args.env, args.seed) 