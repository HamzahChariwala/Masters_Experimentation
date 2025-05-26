#!/bin/bash

# Script to delete evaluation logs and folders from agent directories
# Created on: $(date)

echo "Starting cleanup of evaluation files and folders"

# Set the base directory to search from
BASE_DIR="$(pwd)/Agent_Storage"
echo "Base directory: $BASE_DIR"

# Count variables for tracking
DIRS_REMOVED=0
FILES_REMOVED=0

# Find and delete evaluation_logs directories
echo "Finding and removing evaluation_logs directories..."
for dir in $(find "$BASE_DIR" -type d -name "evaluation_logs"); do
  echo "Removing directory: $dir"
  rm -rf "$dir"
  DIRS_REMOVED=$((DIRS_REMOVED + 1))
done

# Find and delete evaluation_summary directories
echo "Finding and removing evaluation_summary directories..."
for dir in $(find "$BASE_DIR" -type d -name "evaluation_summary"); do
  echo "Removing directory: $dir"
  rm -rf "$dir"
  DIRS_REMOVED=$((DIRS_REMOVED + 1))
done

# Find and delete final_eval.json files
echo "Finding and removing final_eval.json files..."
for file in $(find "$BASE_DIR" -type f -name "final_eval.json"); do
  echo "Removing file: $file"
  rm -f "$file"
  FILES_REMOVED=$((FILES_REMOVED + 1))
done

echo "Cleanup complete!"
echo "Total directories removed: $DIRS_REMOVED"
echo "Total files removed: $FILES_REMOVED" 