#!/bin/bash

# Test script for a single agent
AGENT_GROUP="Standard"
BASE_DIR="LavaTests"

echo "========================================================================================="
echo "Processing category summaries for: ${AGENT_GROUP}"
echo "Base directory: ${BASE_DIR}"
echo "Agent type: ${AGENT_GROUP}"
echo "========================================================================================="

# Run the category_summaries.py script for this agent group
python ./Agent_Evaluation/category_summaries.py --base "${BASE_DIR}" --agent "${AGENT_GROUP}"

echo "Completed category summaries for ${AGENT_GROUP}" 