#!/bin/bash

# Test and verification script for the State Corruption Tool

# Set the output directory
OUTPUT_DIR="test_agent"

# Run the test script to create sample data and launch the tool
echo "Creating sample data and launching corruption tool..."
python Neuron_Selection/CorruptionTooling/test_tool.py --output $OUTPUT_DIR

# After the tool is closed, verify the outputs
echo ""
echo "Verifying outputs..."
python Neuron_Selection/CorruptionTooling/verify_outputs.py --path $OUTPUT_DIR

echo ""
echo "Done! Check the $OUTPUT_DIR directory for the generated files." 