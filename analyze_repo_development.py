#!/usr/bin/env python3
"""
Repository Development Analyzer

This script safely analyzes the development of a Git repository over time by:
1. Finding the last commit made each day
2. Counting lines in Python files for each day's snapshot
3. Tracking which files/folders were worked on
4. Generating data for development timeline charts

SAFETY: This script is completely read-only and will not modify your repository in any way.
It only uses Git's read-only commands (git log, git show, git ls-tree).
"""

import subprocess
import json
import csv
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from pathlib import Path


def run_git_command(command, cwd=None):
    """
    Safely run a read-only Git command and return the output.
    Only allows read-only Git operations.
    """
    # Whitelist of safe Git commands
    safe_commands = ['log', 'show', 'ls-tree', 'rev-list', 'rev-parse']
    
    cmd_parts = command.split()
    if not cmd_parts[0] == 'git' or cmd_parts[1] not in safe_commands:
        raise ValueError(f"Unsafe Git command: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {command}")
        print(f"Error: {e.stderr}")
        return None


def get_commit_dates(repo_path):
    """Get all unique dates when commits were made."""
    print("📅 Analyzing commit history...")
    
    # Get all commits with their dates (removed quotes from format)
    cmd = 'git log --pretty=format:%cd|%H --date=short'
    output = run_git_command(cmd, repo_path)
    
    if not output:
        return []
    
    # Parse dates and commits
    date_commits = {}
    for line in output.split('\n'):
        if '|' in line:
            date_str, commit_hash = line.split('|')
            # Clean any quotes or whitespace
            date_str = date_str.strip().strip('"')
            commit_hash = commit_hash.strip()
            if date_str not in date_commits:
                date_commits[date_str] = []
            date_commits[date_str].append(commit_hash)
    
    # Sort dates and get the last commit of each day
    daily_commits = OrderedDict()
    for date_str in sorted(date_commits.keys()):
        # The first commit in the list is the most recent for that day
        # (because git log shows newest first)
        daily_commits[date_str] = date_commits[date_str][0]
    
    print(f"Found {len(daily_commits)} days with commits")
    return daily_commits


def get_python_files_at_commit(repo_path, commit_hash):
    """Get all Python files present at a specific commit."""
    cmd = f'git ls-tree -r --name-only {commit_hash}'
    output = run_git_command(cmd, repo_path)
    
    if not output:
        return []
    
    # Filter for Python files
    python_files = [
        f for f in output.split('\n')
        if f.endswith('.py') and f.strip()
    ]
    
    return python_files


def count_lines_in_file(repo_path, commit_hash, file_path):
    """Count lines in a specific file at a specific commit."""
    cmd = f'git show {commit_hash}:{file_path}'
    content = run_git_command(cmd, repo_path)
    
    if content is None:
        return 0
    
    # Count non-empty lines
    lines = [line for line in content.split('\n') if line.strip()]
    return len(lines)


def categorize_file(file_path):
    """Categorize a file by its directory structure."""
    path_parts = Path(file_path).parts
    
    if len(path_parts) == 1:
        return "root"
    
    # Use the first directory as the main category
    main_category = path_parts[0]
    
    # Add subcategory if there are multiple levels
    if len(path_parts) > 2:
        return f"{main_category}/{path_parts[1]}"
    else:
        return main_category


def analyze_repository_development(repo_path, output_dir="repo_analysis"):
    """
    Main function to analyze repository development over time.
    """
    print("🔍 Starting repository development analysis...")
    print(f"Repository path: {repo_path}")
    
    # Verify this is a Git repository
    if not os.path.exists(os.path.join(repo_path, '.git')):
        print("❌ Error: Not a Git repository!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get daily commits
    daily_commits = get_commit_dates(repo_path)
    
    if not daily_commits:
        print("❌ No commits found!")
        return
    
    # Analyze each day
    development_data = []
    category_totals = defaultdict(lambda: defaultdict(int))  # category -> date -> line_count
    daily_totals = {}
    
    print("\n📊 Analyzing daily snapshots...")
    
    for i, (date_str, commit_hash) in enumerate(daily_commits.items()):
        print(f"Processing {date_str} ({i+1}/{len(daily_commits)}) - {commit_hash[:8]}")
        
        # Get all Python files at this commit
        python_files = get_python_files_at_commit(repo_path, commit_hash)
        
        date_data = {
            'date': date_str,
            'commit': commit_hash,
            'total_files': len(python_files),
            'total_lines': 0,
            'files': [],
            'categories': defaultdict(int)
        }
        
        # Count lines in each file
        for file_path in python_files:
            line_count = count_lines_in_file(repo_path, commit_hash, file_path)
            category = categorize_file(file_path)
            
            file_info = {
                'path': file_path,
                'category': category,
                'lines': line_count
            }
            
            date_data['files'].append(file_info)
            date_data['total_lines'] += line_count
            date_data['categories'][category] += line_count
            
            # Track category totals over time
            category_totals[category][date_str] = category_totals[category].get(date_str, 0) + line_count
        
        daily_totals[date_str] = date_data['total_lines']
        development_data.append(date_data)
    
    # Save detailed data to JSON
    json_path = os.path.join(output_dir, "development_history.json")
    with open(json_path, 'w') as f:
        json.dump(development_data, f, indent=2, default=str)
    
    # Save summary CSV for easy charting
    csv_path = os.path.join(output_dir, "daily_summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Commit', 'Total_Files', 'Total_Lines'])
        
        for data in development_data:
            writer.writerow([
                data['date'],
                data['commit'][:8],
                data['total_files'],
                data['total_lines']
            ])
    
    # Save category breakdown CSV
    category_csv_path = os.path.join(output_dir, "category_breakdown.csv")
    with open(category_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        all_categories = sorted(set(category for data in development_data for category in data['categories'].keys()))
        header = ['Date'] + all_categories
        writer.writerow(header)
        
        # Data rows
        for data in development_data:
            row = [data['date']]
            for category in all_categories:
                row.append(data['categories'].get(category, 0))
            writer.writerow(row)
    
    # Generate summary report
    report_path = os.path.join(output_dir, "development_report.txt")
    with open(report_path, 'w') as f:
        f.write("🚀 Repository Development Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        first_date = min(daily_commits.keys())
        last_date = max(daily_commits.keys())
        f.write(f"📅 Analysis Period: {first_date} to {last_date}\n")
        f.write(f"📊 Total Days Analyzed: {len(daily_commits)}\n")
        f.write(f"📁 Peak File Count: {max(data['total_files'] for data in development_data)}\n")
        f.write(f"📝 Peak Line Count: {max(data['total_lines'] for data in development_data)}\n\n")
        
        # Most active categories
        f.write("🏆 Most Active Categories (by final line count):\n")
        final_data = development_data[-1]
        sorted_categories = sorted(final_data['categories'].items(), key=lambda x: x[1], reverse=True)
        for category, lines in sorted_categories[:10]:
            f.write(f"  {category}: {lines:,} lines\n")
        
        f.write(f"\n📄 Detailed data saved to:\n")
        f.write(f"  - {json_path} (complete history)\n")
        f.write(f"  - {csv_path} (daily totals)\n")
        f.write(f"  - {category_csv_path} (category breakdown)\n")
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"📊 Use the CSV files to create development timeline charts")
    print(f"📖 Check {report_path} for a summary")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze repository development over time (read-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_repo_development.py
  python analyze_repo_development.py --repo /path/to/repo
  python analyze_repo_development.py --output my_analysis
        """
    )
    
    parser.add_argument(
        '--repo', '-r',
        default='.',
        help='Path to Git repository (default: current directory)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='repo_analysis',
        help='Output directory for analysis results (default: repo_analysis)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path for safety
    repo_path = os.path.abspath(args.repo)
    
    print("🔐 SAFETY GUARANTEE: This script only uses read-only Git commands")
    print("🔐 Your repository will NOT be modified in any way\n")
    
    analyze_repository_development(repo_path, args.output)


if __name__ == "__main__":
    main() 