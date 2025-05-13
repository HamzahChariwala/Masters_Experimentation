#!/usr/bin/env python3
"""
Run the flexible spawn distribution demo.
This script provides a simple command-line interface to run the demo.
"""

import os
import argparse
from demo import demo_spawn_distributions
from test_distributions import (
    test_standalone_distributions,
    test_with_environment,
    test_spawn_positions
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flexible Spawn Distribution Demo and Tests"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run tests instead of the demo"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="Number of samples for spawn position tests"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n===== FLEXIBLE SPAWN DISTRIBUTIONS =====\n")
    
    if args.test:
        print("Running tests...")
        test_standalone_distributions()
        test_with_environment()
        test_spawn_positions(num_samples=args.num_samples)
        print("\nAll tests completed.")
    else:
        print("Running demonstration...")
        demo_spawn_distributions()
        print("\nDemonstration completed.")
    
    print("\nOutput images are available in the outputs/ and demo_outputs/ directories.")
    print("\n=========================================\n")


if __name__ == "__main__":
    main()