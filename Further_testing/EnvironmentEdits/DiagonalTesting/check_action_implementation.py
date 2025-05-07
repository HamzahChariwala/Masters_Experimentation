import os
import sys
import inspect

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

# Import the relevant modules
from EnvironmentEdits.BespokeEdits.ActionSpace import CustomActionWrapper

# Get the source code of the step method
step_method = inspect.getsource(CustomActionWrapper.step)

# Print the relevant implementation
print("=== CustomActionWrapper.step() Implementation ===")
print(step_method)

# Check if we can access the actual class
print("\n=== Examining actual implementation of diagonal moves ===")
try:
    # Create a dummy wrapper to examine
    wrapper = CustomActionWrapper(None)
    
    # Check specific attributes or methods related to diagonal moves
    for name, method in inspect.getmembers(wrapper, inspect.ismethod):
        if "diagonal" in name.lower():
            print(f"\nMethod: {name}")
            print(inspect.getsource(method))
    
    # Check diagonal-related attributes
    for name in dir(wrapper):
        if "diagonal" in name.lower() and not name.startswith("__"):
            print(f"\nAttribute: {name}")
            try:
                value = getattr(wrapper, name)
                print(f"Value: {value}")
            except:
                print("Unable to retrieve value")
except Exception as e:
    print(f"Error examining wrapper: {e}")

print("\n=== Looking for diagonal implementation in source code ===")
try:
    with open("EnvironmentEdits/BespokeEdits/ActionSpace.py", "r") as file:
        content = file.read()
        
        # Find sections related to diagonal movement
        lines = content.split("\n")
        in_diagonal_section = False
        diagonal_lines = []
        
        for i, line in enumerate(lines):
            if "diagonal" in line.lower() or in_diagonal_section:
                if "def " in line and "diagonal" not in line.lower():
                    in_diagonal_section = False
                    continue
                
                if "def " in line and "diagonal" in line.lower():
                    in_diagonal_section = True
                
                if in_diagonal_section:
                    diagonal_lines.append(f"{i+1}: {line}")
        
        print("\nDiagonal movement implementation:")
        print("\n".join(diagonal_lines))
except Exception as e:
    print(f"Error reading source file: {e}") 