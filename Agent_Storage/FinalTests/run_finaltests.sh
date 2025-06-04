#!/bin/bash

# Script to train and evaluate all FinalTests agents
# Run from: Agent_Storage/FinalTests/

echo "========================================================================"
echo "STARTING FINALTESTS TRAINING AND EVALUATION PIPELINE"
echo "========================================================================"

# Get the current directory (should be Agent_Storage/FinalTests)
current_dir=$(pwd)
echo "Current directory: $current_dir"

# Go to the root directory (two levels up)
cd ../../

echo "Moving to root directory: $(pwd)"
echo ""

# Find all FinalTests agent directories
echo "Finding FinalTests agent directories..."
agent_dirs=$(find Agent_Storage/FinalTests -name "config.yaml" -type f -exec dirname {} \; | grep -v "Agent_Storage/FinalTests/config.yaml" | sort)

echo "Found agent directories:"
for dir in $agent_dirs; do
    echo "  - $dir"
done
echo ""

# TRAINING PHASE
echo "========================================================================"
echo "PHASE 1: TRAINING ALL FINALTESTS AGENTS"
echo "========================================================================"

training_count=0
for agent_dir in $agent_dirs; do
    # Extract relative path from Agent_Storage
    relative_path=${agent_dir#"Agent_Storage/"}
    
    training_count=$((training_count + 1))
    echo "[$training_count/10] Training agent: $relative_path"
    echo "Command: python Agent_Training/train.py --path $relative_path"
    
    # Run the training command
    python Agent_Training/train.py --path $relative_path
    
    if [ $? -eq 0 ]; then
        echo "✓ Training completed successfully for $relative_path"
    else
        echo "✗ Training failed for $relative_path"
    fi
    
    echo ""
    
    # Add a small delay between trainings
    sleep 2
done

echo "========================================================================"
echo "PHASE 1 COMPLETE: All training jobs finished!"
echo "========================================================================"
echo ""

# EVALUATION PHASE
echo "========================================================================"
echo "PHASE 2: EVALUATING ALL FINALTESTS AGENTS"
echo "========================================================================"

evaluation_count=0
for agent_dir in $agent_dirs; do
    evaluation_count=$((evaluation_count + 1))
    echo "[$evaluation_count/10] =========================================="
    echo "Evaluating agent: $agent_dir"
    echo "=========================================="
    
    # Run the evaluation command with the full path
    python Agent_Evaluation/generate_evaluations.py --path "$agent_dir"
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully for $agent_dir"
    else
        echo "✗ Evaluation failed for $agent_dir"
    fi
    
    echo ""
done

echo "========================================================================"
echo "PHASE 2 COMPLETE: All evaluations finished!"
echo "========================================================================"
echo ""

# SUMMARY
echo "========================================================================"
echo "FINALTESTS PIPELINE COMPLETE!"
echo "========================================================================"
echo "Trained and evaluated 10 FinalTests agents:"
for agent_dir in $agent_dirs; do
    agent_name=$(basename "$agent_dir")
    echo "  ✓ $agent_name"
done

echo ""
echo "Results can be found in:"
echo "  - Agent_Storage/FinalTests/FinalTests-v*/agent.zip (trained models)"
echo "  - Agent_Storage/FinalTests/FinalTests-v*/evaluation_summary/ (results)"
echo "  - Agent_Storage/FinalTests/FinalTests-v*/final_eval.json (performance)"
echo ""
echo "To view results: navigate to Agent_Storage/FinalTests/ and check individual folders"
echo "========================================================================"

# Return to original directory
cd "$current_dir"
echo "Returned to: $(pwd)" 