"""
Import file for backward compatibility with existing code.
Redirects to the new log.py file.
"""
from typing import Dict, Any

import numpy as np
from Behaviour_Specification.log import analyze_navigation_graphs

# Re-export the analyze_navigation_graphs function for backward compatibility
__all__ = ['analyze_navigation_graphs'] 