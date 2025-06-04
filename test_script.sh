#!/bin/bash

echo "Script starting..."
echo "Working directory: $(pwd)"

# Test one agent only
agent_name="FinalTests-v1"
echo "Testing agent: $agent_name"

# Test the training command
echo "Running: python Agent_Training/train.py --path \"FinalTests/${agent_name}\""
python Agent_Training/train.py --path "FinalTests/${agent_name}"

if [ $? -eq 0 ]; then
    echo "SUCCESS: Training completed"
else
    echo "FAILED: Training failed"
fi

echo "Script finished." 