#!/bin/bash

# Simple script to train and evaluate all FinalTests agents

echo "========================================"
echo "     FinalTests Training Pipeline"
echo "========================================"
echo ""

echo "Working directory: $(pwd)"
echo ""

# Phase 1: Training
echo "Phase 1: Training all agents..."
echo ""

for i in {1..10}; do
    agent_name="FinalTests-v${i}"
    echo "Training agent ${i}/10: ${agent_name}"
    
    # Train the agent
    if python Agent_Training/train.py --path "FinalTests/${agent_name}"; then
        echo "SUCCESS: Training completed for ${agent_name}"
    else
        echo "FAILED: Training failed for ${agent_name}"
    fi
    
    echo ""
done

echo "Phase 1 Complete: All training attempts finished"
echo ""

# Phase 2: Evaluation
echo "Phase 2: Evaluating all agents..."
echo ""

for i in {1..10}; do
    agent_name="FinalTests-v${i}"
    echo "Evaluating agent ${i}/10: ${agent_name}"
    
    # Check if agent.zip exists
    agent_path="Agent_Storage/FinalTests/${agent_name}"
    if [ -f "${agent_path}/agent.zip" ]; then
        # Evaluate the agent
        if python Agent_Evaluation/generate_evaluations.py --path "${agent_path}"; then
            echo "SUCCESS: Evaluation completed for ${agent_name}"
        else
            echo "FAILED: Evaluation failed for ${agent_name}"
        fi
    else
        echo "SKIPPED: agent.zip not found for ${agent_name}"
    fi
    
    echo ""
done

echo "Phase 2 Complete: All evaluation attempts finished"
echo ""

# Summary
echo "========================================"
echo "             Summary Report"
echo "========================================"

trained_count=0
evaluated_count=0

for i in {1..10}; do
    agent_name="FinalTests-v${i}"
    agent_path="Agent_Storage/FinalTests/${agent_name}"
    
    if [ -f "${agent_path}/agent.zip" ]; then
        trained_count=$((trained_count + 1))
        echo "TRAINED: ${agent_name}"
        
        if [ -f "${agent_path}/final_eval.json" ]; then
            evaluated_count=$((evaluated_count + 1))
            echo "  └─ EVALUATED"
        else
            echo "  └─ NOT EVALUATED"
        fi
    else
        echo "NOT TRAINED: ${agent_name}"
    fi
done

echo ""
echo "Training: ${trained_count}/10 agents successfully trained"
echo "Evaluation: ${evaluated_count}/10 agents successfully evaluated"
echo ""
echo "FinalTests pipeline complete!" 