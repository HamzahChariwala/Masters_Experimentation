# Repository Development Analysis

This tool provides a comprehensive, **completely safe** way to analyze how your Git repository has developed over time. It tracks Python file growth, categorizes code by directory structure, and generates beautiful visualizations showing your development patterns.

## 🔐 Safety Guarantee

**These scripts are 100% READ-ONLY and will NEVER modify your repository.** They only use Git's read-only commands (`git log`, `git show`, `git ls-tree`) to extract historical data without making any changes to your codebase.

## 📋 Requirements

```bash
pip install pandas matplotlib seaborn numpy
```

## 🚀 Quick Start

### Step 1: Analyze Your Repository

```bash
# Analyze current directory (if it's a Git repo)
python analyze_repo_development.py

# Or specify a different repository
python analyze_repo_development.py --repo /path/to/your/repo

# Customize output directory
python analyze_repo_development.py --output my_analysis
```

### Step 2: Generate Visualizations

```bash
# Generate charts from the analysis
python create_development_charts.py

# Or specify custom directories
python create_development_charts.py --analysis-dir my_analysis --output-dir my_charts
```

## 📊 What You Get

### Analysis Data

The analyzer creates several data files:

- **`development_history.json`** - Complete detailed history with file-by-file breakdown
- **`daily_summary.csv`** - Daily totals (perfect for spreadsheet analysis)
- **`category_breakdown.csv`** - Lines of code by category over time
- **`development_report.txt`** - Human-readable summary

### Beautiful Charts

The chart generator creates 6 different visualizations:

1. **Overall Growth** - Total lines and files over time
2. **Category Evolution** - Stacked area chart showing code distribution
3. **Development Velocity** - Daily changes (green for additions, red for removals)
4. **Activity Heatmap** - Which categories you worked on each day
5. **File Size Distribution** - Current file sizes and distribution by category
6. **Development Dashboard** - Comprehensive overview with key statistics

## 📁 How It Works

### File Categorization

Files are automatically categorized by their directory structure:

- `root` - Files in the main directory
- `Visualisation_Tools` - Files in this folder
- `Behaviour_Specification/StateGeneration` - Nested folder structure
- And so on...

### Daily Snapshots

For each day with commits, the tool:

1. Finds the **last commit** of that day
2. Counts lines in **all Python files** at that point
3. Categorizes files by their folder structure
4. Tracks which areas of the codebase were active

### Line Counting

- Only counts **non-empty lines** (ignores blank lines)
- Only analyzes **Python files** (`.py` extension)
- Uses the actual file content from each historical commit

## 📈 Example Use Cases

- **Track project growth** - See how your codebase expanded over time
- **Identify focus areas** - Which parts of your project received the most attention
- **Development patterns** - Understand your coding rhythm and productivity
- **Project milestones** - Correlate code growth with project phases
- **Team analysis** - See overall development velocity
- **Portfolio documentation** - Professional charts for showcasing your work

## 🎨 Customization

### Analyze Specific Date Ranges

You can modify the Git commands in `analyze_repo_development.py` to focus on specific periods:

```python
# In get_commit_dates function, modify the command:
cmd = 'git log --since="2023-01-01" --until="2023-12-31" --pretty=format:"%cd|%H" --date=short'
```

### Different File Types

To analyze other file types besides Python:

```python
# In get_python_files_at_commit function, modify the filter:
files = [f for f in output.split('\n') if f.endswith('.js') and f.strip()]  # For JavaScript
```

### Custom Categories

Modify the `categorize_file` function to create custom category logic:

```python
def categorize_file(file_path):
    if 'test' in file_path.lower():
        return "tests"
    elif 'model' in file_path.lower():
        return "models"
    # ... your custom logic
```

## 🔍 Troubleshooting

### "Not a Git repository"

Make sure you're running the script in a directory that contains a `.git` folder, or use the `--repo` flag to specify the correct path.

### "No commits found"

This usually means:
- The repository is empty
- You're not in the right directory
- There's an issue with Git command execution

### Missing Dependencies

Install required packages:
```bash
pip install pandas matplotlib seaborn numpy
```

### Charts Look Weird

If dates are overlapping or charts look crowded:
- The script automatically adjusts for different repository sizes
- For very active repositories, consider analyzing specific date ranges
- You can modify the date formatting in the chart functions

## 📊 Sample Output

After running both scripts, you'll have:

```
repo_analysis/
├── development_history.json      # Complete data
├── daily_summary.csv            # For spreadsheets
├── category_breakdown.csv       # Category trends
└── development_report.txt       # Summary

charts/
├── overall_growth.png           # Growth trends
├── category_evolution.png       # Code distribution
├── development_velocity.png     # Daily changes
├── category_heatmap.png        # Activity patterns
├── file_size_distribution.png  # File statistics
└── development_dashboard.png   # Complete overview
```

## 🤝 Contributing

Feel free to extend these scripts for your specific needs! Some ideas:

- Add support for other file types
- Include commit message analysis
- Add author-specific breakdowns
- Create interactive plots with Plotly
- Add code complexity metrics

---

**Remember: These scripts are completely safe and read-only. They will never modify your repository!** 🔐 