#!/usr/bin/env python3
"""
Configuration Diagnostic for Grok4Trades

This script helps diagnose configuration loading issues by:
1. Tracing how run_minimal.py loads its configuration
2. Finding all config files in the project
3. Checking parameter mappings and values
4. Identifying the actual config being used
"""

import os
import sys
import re
import json
import yaml
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ANSI color codes for better output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

class ConfigDiagnostics:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.config_files = []
        self.risk_params = {}
        self.config_sources = {}
        self.issues = []
    
    def print_header(self, text: str) -> None:
        """Print a formatted header"""
        print(f"\n{BOLD}{'='*80}\n{text}\n{'='*80}{RESET}")
    
    def print_warning(self, text: str) -> None:
        """Print a warning message"""
        print(f"{YELLOW}⚠️  {text}{RESET}")
    
    def print_error(self, text: str) -> None:
        """Print an error message"""
        print(f"{RED}❌ {text}{RESET}")
    
    def print_success(self, text: str) -> None:
        """Print a success message"""
        print(f"{GREEN}✅ {text}{RESET}")
    
    def find_config_files(self) -> List[str]:
        """Find all config files in the project"""
        self.print_header("🔍 SEARCHING FOR CONFIG FILES")
        
        config_patterns = [
            '**/config*.json',
            '**/config*.yaml',
            '**/config*.yml',
            '**/settings*.json',
            '**/settings*.yaml',
            '**/settings*.yml',
            '**/*.env',
            '.env',
            'config.py'
        ]
        
        found_files = []
        for pattern in config_patterns:
            for path in self.project_root.glob(pattern):
                if path.is_file() and path not in found_files:
                    found_files.append(str(path))
        
        self.config_files = sorted(found_files)
        
        if not self.config_files:
            self.print_warning("No configuration files found!")
        else:
            print("Found configuration files:")
            for i, path in enumerate(self.config_files, 1):
                print(f"  {i}. {path}")
        
        return self.config_files
    
    def analyze_run_minimal(self) -> Dict[str, Any]:
        """Analyze run_minimal.py to understand config loading"""
        self.print_header("🔍 ANALYZING run_minimal.py")
        
        run_minimal_path = self.project_root / 'run_minimal.py'
        if not run_minimal_path.exists():
            self.print_error("run_minimal.py not found!")
            return {}
        
        # Read the file content
        with open(run_minimal_path, 'r') as f:
            content = f.read()
        
        # Look for config loading patterns
        config_loading = {
            'imports': [],
            'config_loading': [],
            'risk_params': {}
        }
        
        # Check for common config loading patterns
        if 'import config' in content or 'from config import' in content:
            config_loading['imports'].append("Uses a config.py file")
        
        if 'json.load' in content:
            config_loading['config_loading'].append("Loads config from JSON file")
        
        if 'yaml.safe_load' in content or 'yaml.load' in content:
            config_loading['config_loading'].append("Loads config from YAML file")
        
        # Look for risk manager initialization
        risk_manager_match = re.search(
            r'BasicRiskManager\s*\(([^)]*)\)',
            content,
            re.DOTALL
        )
        
        if risk_manager_match:
            params_str = risk_manager_match.group(1)
            print(f"Found BasicRiskManager initialization with params: {params_str}")
            
            # Try to extract named parameters
            named_params = re.findall(r'(\w+)\s*=', params_str)
            if named_params:
                config_loading['risk_params'] = {p: f"Found in BasicRiskManager call" for p in named_params}
                print(f"Named parameters: {', '.join(named_params)}")
        
        # Print findings
        if not config_loading['imports'] and not config_loading['config_loading']:
            print("Could not determine config loading mechanism from run_minimal.py")
        else:
            if config_loading['imports']:
                print("\nImports detected:", ", ".join(config_loading['imports']))
            if config_loading['config_loading']:
                print("Config loading detected:", ", ".join(config_loading['config_loading']))
            if config_loading['risk_params']:
                print("\nRisk manager parameters:")
                for param, source in config_loading['risk_params'].items():
                    print(f"  - {param}: {source}")
        
        return config_loading
    
    def check_risk_manager_params(self) -> Dict[str, Any]:
        """Check BasicRiskManager parameter requirements"""
        self.print_header("🔍 ANALYZING BasicRiskManager")
        
        # Try to import BasicRiskManager to inspect its parameters
        try:
            # Dynamically import the module containing BasicRiskManager
            risk_manager_path = self.project_root / 'risk' / 'basic_limits.py'
            if not risk_manager_path.exists():
                self.print_warning(f"Could not find risk manager at {risk_manager_path}")
                return {}
            
            spec = importlib.util.spec_from_file_location("risk_manager", risk_manager_path)
            risk_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(risk_module)
            
            # Get the BasicRiskManager class
            if not hasattr(risk_module, 'BasicRiskManager'):
                self.print_error("Could not find BasicRiskManager class in risk manager module")
                return {}
            
            # Get the __init__ method parameters
            init_signature = inspect.signature(risk_module.BasicRiskManager.__init__)
            params = list(init_signature.parameters.values())
            
            # Skip 'self' parameter
            params = [p for p in params if p.name != 'self']
            
            print("BasicRiskManager parameters:")
            for param in params:
                default = f" (default: {param.default})" if param.default != inspect.Parameter.empty else ""
                print(f"  - {param.name}: {param.annotation}{default}")
            
            return {
                'parameters': {p.name: str(p.annotation) for p in params},
                'required': [p.name for p in params if p.default == inspect.Parameter.empty]
            }
            
        except Exception as e:
            self.print_error(f"Error analyzing BasicRiskManager: {e}")
            return {}
    
    def check_config_values(self) -> None:
        """Check config files for risk parameters"""
        self.print_header("🔍 CHECKING CONFIG FILES")
        
        if not self.config_files:
            self.print_warning("No config files found to check")
            return
        
        for config_file in self.config_files:
            print(f"\nChecking: {config_file}")
            
            try:
                if config_file.endswith(('.yaml', '.yml')):
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f) or {}
                elif config_file.endswith('.json'):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                else:
                    continue
                
                # Check for risk parameters
                risk_params = config.get('risk', {}) if isinstance(config, dict) else {}
                
                if not risk_params:
                    print("  No 'risk' section found")
                    continue
                
                print("  Risk parameters found:")
                for key, value in risk_params.items():
                    print(f"    {key}: {value}")
                    
                    # Check for max_daily_loss specifically
                    if key == 'max_daily_loss' and not (0 <= float(value) <= 0.1):
                        self.print_error(f"    ❌ Invalid {key}: {value} (must be between 0 and 0.1)")
                    
                    # Store for summary
                    self.risk_params[key] = value
                
            except Exception as e:
                self.print_error(f"  Error reading {config_file}: {e}")
    
    def suggest_fix(self) -> None:
        """Suggest how to fix the configuration"""
        self.print_header("🔧 SUGGESTED FIX")
        
        if 'max_daily_loss' in self.risk_params:
            current = self.risk_params['max_daily_loss']
            if not (0 <= float(current) <= 0.1):
                print(f"1. Update max_daily_loss to a value between 0 and 0.1 (current: {current})")
                print(f"   Example: 'max_daily_loss': 0.02  # 2% maximum daily loss")
        else:
            print("1. Add max_daily_loss to your risk configuration")
            print("   Example config.json:")
            print('   {
     "risk": {
       "max_daily_loss": 0.02,
       "max_position_size": 0.1,
       "stop_loss_pct": 0.02,
       "take_profit_pct": 0.04
     }
   }')
        
        print("\n2. Ensure your config file is in one of these locations:")
        for path in ['config.json', 'config/config.json', 'settings.json']:
            print(f"   - {self.project_root}/{path}")
        
        print("\n3. If using a different config location, specify it when running the bot:")
        print(f"   python run_minimal.py --config /path/to/your/config.json")
    
    def run_diagnostics(self) -> None:
        """Run all diagnostic checks"""
        self.find_config_files()
        self.analyze_run_minimal()
        self.check_risk_manager_params()
        self.check_config_values()
        self.suggest_fix()
        
        self.print_header("✅ DIAGNOSTIC COMPLETE")
        print("\nNext steps:")
        print("1. Review the findings above")
        print("2. Update your configuration file with the correct parameters")
        print("3. Run the bot with: python run_minimal.py")
        print("4. If issues persist, run: python diagnose_config.py --verbose")

def main():
    """Main function"""
    print(f"{BOLD}🔧 Grok4Trades Configuration Diagnostics{RESET}\n")
    
    diag = ConfigDiagnostics()
    diag.run_diagnostics()

if __name__ == "__main__":
    main()
