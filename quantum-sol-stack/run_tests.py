#!/usr/bin/env python3
"""
SniperBot Test Runner

This script provides a convenient way to run the SniperBot test suite
with different configurations.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_type, network='goerli', verbose=False):
    """Run the test suite with the specified configuration.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        network: Network to test against ('goerli', 'sepolia', 'local')
        verbose: Whether to show verbose test output
    """
    # Set environment variables based on network
    env = os.environ.copy()
    
    if network == 'local':
        env['USE_ETHEREUM_TESTER'] = 'true'
    
    # Build the pytest command
    cmd = [
        'pytest',
        '-xvs' if verbose else '-x',
        '--cov=sniper_bot',
        '--cov-report=term-missing',
    ]
    
    # Add test markers based on test type
    if test_type == 'unit':
        cmd.extend(['-m', 'not integration'])
    elif test_type == 'integration':
        cmd.extend(['-m', 'integration'])
    
    # Add the tests directory
    cmd.append('tests/')
    
    # Print the test configuration
    print(f"\n{'='*50}")
    print(f"Running {test_type} tests on {network} network")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, env=env, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run SniperBot tests')
    parser.add_argument(
        '--test-type',
        choices=['unit', 'integration', 'all'],
        default='unit',
        help='Type of tests to run (default: unit)'
    )
    parser.add_argument(
        '--network',
        choices=['goerli', 'sepolia', 'local'],
        default='goerli',
        help='Network to test against (default: goerli)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Verify .env exists for non-local tests
    if args.network != 'local' and not Path('.env').exists():
        print("Error: .env file not found. Create one from .env.example")
        sys.exit(1)
    
    run_tests(args.test_type, args.network, args.verbose)

if __name__ == '__main__':
    main()
