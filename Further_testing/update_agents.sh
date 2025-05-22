#!/bin/bash

# Script to update agents:
# 1. Delete agent.zip files
# 2. Reduce training timesteps in config.yaml by 90%

echo "Starting agent update process..."

# Find all agent.zip files and delete them
echo "Deleting all agent.zip files..."
find Agent_Storage -name "agent.zip" -type f -exec rm {} \;

# Find all config.yaml files and update timesteps (reduce by 90%)
echo "Updating config.yaml files to reduce timesteps by 90%..."

find Agent_Storage -name "config.yaml" -type f | while read -r config_file; do
  echo "Processing $config_file"
  
  # Create a temporary file
  temp_file=$(mktemp)
  
  # Use sed to reduce total_timesteps by 90%
  sed -E 's/(total_timesteps:[ ]*)[0-9_]+/\1'"$(awk 'BEGIN {print int(5000000*0.1)}')"'/' "$config_file" > "$temp_file"
  
  # Also update check_freq in evaluation section (reduce by 90% if present)
  sed -i '' -E 's/(check_freq:[ ]*)[0-9_]+/\1'"$(awk 'BEGIN {print int(500000*0.1)}')"'/' "$temp_file"
  
  # Replace the original file with the modified one
  mv "$temp_file" "$config_file"
  
  echo "Updated $config_file"
done

echo "Agent update process complete!" 