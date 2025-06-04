#!/usr/bin/env python3
"""
Development Charts Generator

This script creates beautiful visualizations from the repository development analysis data.
It generates various charts showing how your codebase evolved over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path


def load_analysis_data(analysis_dir):
    """Load the analysis data from the specified directory."""
    
    # Load detailed JSON data
    json_path = os.path.join(analysis_dir, "development_history.json")
    with open(json_path, 'r') as f:
        detailed_data = json.load(f)
    
    # Load CSV data
    daily_summary = pd.read_csv(os.path.join(analysis_dir, "daily_summary.csv"))
    category_breakdown = pd.read_csv(os.path.join(analysis_dir, "category_breakdown.csv"))
    
    # Convert dates
    daily_summary['Date'] = pd.to_datetime(daily_summary['Date'])
    category_breakdown['Date'] = pd.to_datetime(category_breakdown['Date'])
    
    return detailed_data, daily_summary, category_breakdown


def create_overall_growth_chart(daily_summary, output_dir):
    """Create a chart showing overall codebase growth over time."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Total lines over time
    ax1.plot(daily_summary['Date'], daily_summary['Total_Lines'], 
             linewidth=2.5, color='#2E86AB', marker='o', markersize=3)
    ax1.fill_between(daily_summary['Date'], daily_summary['Total_Lines'], 
                     alpha=0.3, color='#2E86AB')
    ax1.set_ylabel('Total Lines of Code', fontsize=12)
    ax1.set_title('Repository Growth Over Time', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Number of files over time
    ax2.plot(daily_summary['Date'], daily_summary['Total_Files'], 
             linewidth=2.5, color='#A23B72', marker='s', markersize=3)
    ax2.fill_between(daily_summary['Date'], daily_summary['Total_Files'], 
                     alpha=0.3, color='#A23B72')
    ax2.set_ylabel('Number of Python Files', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_growth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📈 Created overall growth chart")


def create_category_evolution_chart(category_breakdown, output_dir):
    """Create a stacked area chart showing how different categories evolved."""
    
    # Prepare data
    date_col = category_breakdown['Date']
    category_cols = [col for col in category_breakdown.columns if col != 'Date']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create a color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_cols)))
    
    # Create stacked area plot
    ax.stackplot(date_col, *[category_breakdown[col] for col in category_cols], 
                labels=category_cols, colors=colors, alpha=0.8)
    
    ax.set_title('Code Distribution by Category Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Lines of Code', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Created category evolution chart")


def create_development_velocity_chart(daily_summary, output_dir):
    """Create a chart showing development velocity (changes per day)."""
    
    # Calculate daily changes
    daily_summary = daily_summary.copy()
    daily_summary['Lines_Change'] = daily_summary['Total_Lines'].diff()
    daily_summary['Files_Change'] = daily_summary['Total_Files'].diff()
    
    # Remove the first row (NaN values)
    daily_summary = daily_summary.iloc[1:]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Lines added/removed per day
    colors = ['#28A745' if x >= 0 else '#DC3545' for x in daily_summary['Lines_Change']]
    ax1.bar(daily_summary['Date'], daily_summary['Lines_Change'], 
            color=colors, alpha=0.7, width=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Lines Changed', fontsize=12)
    ax1.set_title('Development Velocity', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Files added/removed per day
    colors = ['#007BFF' if x >= 0 else '#FD7E14' for x in daily_summary['Files_Change']]
    ax2.bar(daily_summary['Date'], daily_summary['Files_Change'], 
            color=colors, alpha=0.7, width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Files Changed', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'development_velocity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("⚡ Created development velocity chart")


def create_category_heatmap(category_breakdown, output_dir):
    """Create a heatmap showing activity in different categories over time."""
    
    # Prepare data for heatmap
    date_col = category_breakdown['Date']
    category_cols = [col for col in category_breakdown.columns if col != 'Date']
    
    # Calculate daily changes for each category
    heatmap_data = []
    for i in range(1, len(category_breakdown)):
        daily_changes = []
        for col in category_cols:
            change = category_breakdown.iloc[i][col] - category_breakdown.iloc[i-1][col]
            daily_changes.append(change)
        heatmap_data.append(daily_changes)
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(heatmap_data, 
                             columns=category_cols,
                             index=date_col.iloc[1:])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.heatmap(heatmap_df.T, cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Lines Added/Removed'},
                xticklabels=False, ax=ax)
    
    ax.set_title('Development Activity Heatmap by Category', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("🔥 Created category activity heatmap")


def create_file_size_distribution(detailed_data, output_dir):
    """Create charts showing file size distributions over time."""
    
    # Get file sizes for the most recent day
    latest_data = detailed_data[-1]
    file_sizes = [file_info['lines'] for file_info in latest_data['files']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram of file sizes
    ax1.hist(file_sizes, bins=20, alpha=0.7, color='#17A2B8', edgecolor='black')
    ax1.set_xlabel('Lines per File', fontsize=12)
    ax1.set_ylabel('Number of Files', fontsize=12)
    ax1.set_title('File Size Distribution (Current)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Box plot of file sizes by category
    category_data = {}
    for file_info in latest_data['files']:
        category = file_info['category']
        if category not in category_data:
            category_data[category] = []
        category_data[category].append(file_info['lines'])
    
    categories = list(category_data.keys())
    sizes = [category_data[cat] for cat in categories]
    
    bp = ax2.boxplot(sizes, labels=categories, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Lines per File', fontsize=12)
    ax2.set_title('File Size Distribution by Category', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'file_size_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📏 Created file size distribution charts")


def create_summary_dashboard(daily_summary, category_breakdown, output_dir):
    """Create a comprehensive dashboard with key metrics."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Total lines trend (large)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(daily_summary['Date'], daily_summary['Total_Lines'], 
             linewidth=3, color='#2E86AB', marker='o')
    ax1.fill_between(daily_summary['Date'], daily_summary['Total_Lines'], alpha=0.3, color='#2E86AB')
    ax1.set_title('Total Lines of Code', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. File count trend (large)
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(daily_summary['Date'], daily_summary['Total_Files'], 
             linewidth=3, color='#A23B72', marker='s')
    ax2.fill_between(daily_summary['Date'], daily_summary['Total_Files'], alpha=0.3, color='#A23B72')
    ax2.set_title('Number of Files', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Category breakdown (current state)
    ax3 = fig.add_subplot(gs[1, :2])
    latest_categories = category_breakdown.iloc[-1]
    categories = [col for col in latest_categories.index if col != 'Date' and latest_categories[col] > 0]
    sizes = [latest_categories[col] for col in categories]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    wedges, texts, autotexts = ax3.pie(sizes, labels=categories, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax3.set_title('Current Code Distribution', fontweight='bold')
    
    # 4. Growth rate
    ax4 = fig.add_subplot(gs[1, 2:])
    total_days = len(daily_summary)
    if total_days > 1:
        growth_rate = (daily_summary['Total_Lines'].iloc[-1] - daily_summary['Total_Lines'].iloc[0]) / total_days
        ax4.bar(['Average Daily Growth'], [growth_rate], color='#28A745', alpha=0.7)
        ax4.set_title('Average Daily Growth', fontweight='bold')
        ax4.set_ylabel('Lines per Day')
    
    # 5. Key statistics (bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate statistics
    total_lines = daily_summary['Total_Lines'].iloc[-1]
    total_files = daily_summary['Total_Files'].iloc[-1]
    days_active = len(daily_summary)
    avg_file_size = total_lines / total_files if total_files > 0 else 0
    
    stats_text = f"""
    📊 Repository Statistics
    
    📝 Total Lines: {total_lines:,}
    📁 Total Files: {total_files:,}
    📅 Days with Commits: {days_active:,}
    📏 Average File Size: {avg_file_size:.1f} lines
    
    📈 Growth: {total_lines / days_active:.1f} lines/day average
    """
    
    ax5.text(0.5, 0.5, stats_text, transform=ax5.transAxes, 
             fontsize=14, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Format dates for all time-based axes
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('Repository Development Dashboard', fontsize=20, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'development_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📋 Created comprehensive dashboard")


def main():
    """Main function to generate all charts."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate development charts from analysis data")
    parser.add_argument('--analysis-dir', '-a', default='repo_analysis',
                       help='Directory containing analysis results')
    parser.add_argument('--output-dir', '-o', default='charts',
                       help='Directory to save charts')
    
    args = parser.parse_args()
    
    # Check if analysis data exists
    if not os.path.exists(args.analysis_dir):
        print(f"❌ Analysis directory not found: {args.analysis_dir}")
        print("Run analyze_repo_development.py first to generate the data.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("📊 Loading analysis data...")
    try:
        detailed_data, daily_summary, category_breakdown = load_analysis_data(args.analysis_dir)
    except Exception as e:
        print(f"❌ Error loading analysis data: {e}")
        return
    
    print("🎨 Generating charts...")
    
    # Generate all charts
    create_overall_growth_chart(daily_summary, args.output_dir)
    create_category_evolution_chart(category_breakdown, args.output_dir)
    create_development_velocity_chart(daily_summary, args.output_dir)
    create_category_heatmap(category_breakdown, args.output_dir)
    create_file_size_distribution(detailed_data, args.output_dir)
    create_summary_dashboard(daily_summary, category_breakdown, args.output_dir)
    
    print(f"\n✅ All charts generated successfully!")
    print(f"📁 Charts saved to: {args.output_dir}/")
    print("\n📊 Generated charts:")
    print("  - overall_growth.png: Total lines and files over time")
    print("  - category_evolution.png: Stacked area chart of code distribution")
    print("  - development_velocity.png: Daily changes in lines and files")
    print("  - category_heatmap.png: Activity heatmap by category")
    print("  - file_size_distribution.png: File size statistics")
    print("  - development_dashboard.png: Comprehensive overview")


if __name__ == "__main__":
    main() 