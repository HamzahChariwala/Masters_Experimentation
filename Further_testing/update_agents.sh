#!/bin/bash

# Script to update agents:
# 1. Delete agent.zip files
# 2. Reduce training timesteps in config.yaml by 80%
# 3. Set learning_starts to 50,000

echo "Starting agent update process..."

# Find all agent.zip files and delete them
echo "Deleting all agent.zip files..."
find Agent_Storage -name "agent.zip" -type f -exec rm {} \;

echo "Deleting all logs files..."
find Agent_Storage -name "logs" -type d -exec rm -rf {} \;

# Find all config.yaml files and update timesteps (reduce by 80%)
echo "Updating config.yaml files to reduce timesteps by 80%..."

find Agent_Storage -name "config.yaml" -type f | while read -r config_file; do
  echo "Processing $config_file"
  
  # Create a temporary file
  temp_file=$(mktemp)
  
  # Use sed to reduce total_timesteps by 80%
  sed -E 's/(total_timesteps:[ ]*)[0-9_]+/\1'"$(awk 'BEGIN {print int(5000000*0.2)}')"'/' "$config_file" > "$temp_file"
  
  # Also update check_freq in evaluation section (reduce by 80% if present)
  sed -i '' -E 's/(check_freq:[ ]*)[0-9_]+/\1'"$(awk 'BEGIN {print int(500000*0.2)}')"'/' "$temp_file"
  
  # Update learning_starts to 50,000
  sed -i '' -E 's/(learning_starts:[ ]*)[0-9_]+/\150000/' "$temp_file"
  
  # Replace the original file with the modified one
  mv "$temp_file" "$config_file"
  
  echo "Updated $config_file"
done

echo "Agent update process complete!"