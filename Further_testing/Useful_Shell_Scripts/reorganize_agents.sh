#!/bin/bash

# Function to process a single agent folder
process_agent_folder() {
    local agent_path="$1"
    local agent_name=$(basename "$agent_path")
    
    echo "Processing agent: $agent_name in $agent_path"
    
    # Create v1 folder and move all contents there
    mkdir -p "$agent_path/$agent_name-v1"
    # Move all files and folders except the new v1 folder
    find "$agent_path" -maxdepth 1 -mindepth 1 ! -name "$agent_name-v1" -exec mv {} "$agent_path/$agent_name-v1" \;
    
    # Create v2-v10 folders and copy config.yaml
    for i in {2..10}; do
        mkdir -p "$agent_path/$agent_name-v$i"
        if [ -f "$agent_path/$agent_name-v1/config.yaml" ]; then
            cp "$agent_path/$agent_name-v1/config.yaml" "$agent_path/$agent_name-v$i/"
        else
            echo "Warning: config.yaml not found in $agent_path/$agent_name-v1"
        fi
    done
}

# Function to find and process all agent folders
find_and_process_agents() {
    local root_dir="$1"
    
    # Find all folders that contain agent.zip (this indicates it's an agent folder)
    while IFS= read -r -d '' agent_zip; do
        agent_dir=$(dirname "$agent_zip")
        process_agent_folder "$agent_dir"
    done < <(find "$root_dir" -name "agent.zip" -print0)
}

# Main execution
echo "Starting agent folder reorganization..."

# Process agents in Agent_Storage
if [ -d "Agent_Storage" ]; then
    find_and_process_agents "Agent_Storage"
    echo "Completed processing Agent_Storage"
else
    echo "Error: Agent_Storage directory not found"
    exit 1
fi

echo "All done!" 