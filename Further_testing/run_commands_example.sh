#!/bin/bash

# This is a shell script to run commands sequentially
# Usage: bash run_commands.sh OR ./run_commands.sh

# Exit script if any command fails
set -e

# Print each command before executing it
set -x

echo "Starting command sequence..."

# Define variables for agent paths
# This demonstrates how you can organize agents in subdirectories
AGENT_DIR="experiments/lava_crossing"
AGENT_NAME="agent_v1"
AGENT_PATH="${AGENT_DIR}/${AGENT_NAME}"
FULL_PATH="Agent_Storage/${AGENT_PATH}"

# Create nested directory structure
mkdir -p ${FULL_PATH}

# Copy the example config for this agent
cp Agent_Storage/example_config.yaml ${FULL_PATH}/config.yaml

# With our changes to train.py, logs will be stored in:
# Agent_Storage/experiments/lava_crossing/agent_v1/logs
echo "Training will store logs in ${FULL_PATH}/logs"

# Run the training script with the path relative to Agent_Storage
# The agent path is 'experiments/lava_crossing/agent_v1'
echo "Starting training process..."
python Agent_Training/train.py --path ${AGENT_PATH}

# After training, the model should be saved to:
# Agent_Storage/experiments/lava_crossing/agent_v1/agent.zip
echo "Checking if training produced agent.zip file..."
ls -la ${FULL_PATH}

# Extract model components - the path can be the agent.zip path
# The extracted model will be saved in:
# Agent_Storage/experiments/lava_crossing/agent_v1/extracted_model
echo "Extracting model components..."
python Inference_Estimation/ModelAnalysis/extract_agent_model.py --path ${FULL_PATH}/agent.zip

# Run performance evaluation on the trained agent
echo "Running performance evaluation..."
python Agent_Training/performance_logging.py --path ${FULL_PATH}/agent.zip

echo "All commands completed successfully!"
echo "All logs and models are stored in ${FULL_PATH}"

# You can also reference the logs directory directly
echo "Training logs are in: ${FULL_PATH}/logs"
echo "Extracted model is in: ${FULL_PATH}/extracted_model" 