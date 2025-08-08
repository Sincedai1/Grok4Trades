#!/usr/bin/env python3
"""
Risk Configuration Fixer for Grok4Trades

This script helps identify and fix risk configuration issues that are preventing
the trading bot from starting. It specifically addresses the max_daily_loss
validation error and ensures all risk parameters are properly configured.
"""

import os
import sys
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Default configuration paths to check
DEFAULT_CONFIG_PATHS = [
    "config.json",
    "config.yaml",
    "config.yml",
    "config/config.json",
    "config/config.yaml",
    "config/config.yml",
    "settings.json",
    "settings.yaml",
    "settings.yml",
]

# Default risk parameters
DEFAULT_RISK_CONFIG = {
    "max_daily_loss": 0.02,  # 2% max daily loss
    "max_position_size": 0.1,  # 10% of portfolio
    "max_leverage": 1.0,  # 1x leverage
    "stop_loss_pct": 0.02,  # 2% stop loss
    "take_profit_pct": 0.04,  # 4% take profit
    "max_drawdown": 0.1,  # 10% max drawdown
    "risk_per_trade": 0.01,  # 1% risk per trade
}

class ConfigFixer:
    """Handles detection and fixing of risk configuration issues"""
    
    def __init__(self):
        self.config_path = None
        self.config_data = {}
        self.config_format = None  # 'json' or 'yaml'
        self.changes_made = False
    
    def find_config_file(self) -> Optional[str]:
        """Search for configuration files in common locations"""
        print("\n🔍 Searching for configuration files...")
        
        for path in DEFAULT_CONFIG_PATHS:
            if os.path.exists(path):
                print(f"✅ Found config file: {path}")
                return path
        
        print("⚠️  No existing config files found. Will create a new one.")
        return None
    
    def load_config(self, path: str) -> bool:
        """Load configuration from file"""
        self.config_path = path
        
        try:
            with open(path, 'r') as f:
                if path.endswith(('.yaml', '.yml')):
                    self.config_data = yaml.safe_load(f) or {}
                    self.config_format = 'yaml'
                else:  # Assume JSON
                    self.config_data = json.load(f)
                    self.config_format = 'json'
            return True
        except Exception as e:
            print(f"❌ Error loading config file {path}: {e}")
            return False
    
    def create_default_config(self) -> bool:
        """Create a new default configuration file"""
        print("\n📝 Creating default configuration...")
        
        # Create config directory if it doesn't exist
        os.makedirs('config', exist_ok=True)
        self.config_path = 'config/config.json'
        self.config_format = 'json'
        
        # Initialize with default risk config
        self.config_data = {
            "exchange": {
                "api_key": "YOUR_API_KEY",
                "api_secret": "YOUR_API_SECRET",
                "testnet": True
            },
            "trading": {
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "initial_balance": 10000.0
            },
            "risk": DEFAULT_RISK_CONFIG
        }
        
        return self.save_config()
    
    def backup_config(self) -> bool:
        """Create a backup of the current config"""
        if not self.config_path:
            return False
            
        backup_path = f"{self.config_path}.bak"
        try:
            shutil.copy2(self.config_path, backup_path)
            print(f"✅ Created backup at: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        if not self.config_path:
            return False
            
        try:
            with open(self.config_path, 'w') as f:
                if self.config_format == 'yaml':
                    yaml.dump(self.config_data, f, default_flow_style=False, sort_keys=False)
                else:  # JSON
                    json.dump(self.config_data, f, indent=2)
            print(f"✅ Configuration saved to: {self.config_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def fix_risk_config(self) -> bool:
        """Fix risk configuration parameters"""
        print("\n🔧 Checking risk configuration...")
        
        # Initialize risk section if it doesn't exist
        if 'risk' not in self.config_data:
            self.config_data['risk'] = {}
            self.changes_made = True
            print("ℹ️  Added missing 'risk' section to config")
        
        risk_config = self.config_data['risk']
        
        # Check and fix each parameter
        for param, default_value in DEFAULT_RISK_CONFIG.items():
            if param not in risk_config:
                risk_config[param] = default_value
                self.changes_made = True
                print(f"ℹ️  Added missing parameter: {param} = {default_value}")
            
            # Special handling for max_daily_loss
            if param == 'max_daily_loss':
                current_value = risk_config[param]
                if not (0 <= current_value <= 0.1):
                    print(f"⚠️  Invalid {param} value: {current_value}. Must be between 0 and 0.1")
                    risk_config[param] = default_value
                    self.changes_made = True
                    print(f"✅ Fixed {param}: {current_value} -> {default_value}")
        
        return True
    
    def run(self):
        """Run the configuration fixer"""
        print("\n" + "="*60)
        print("🔧 Grok4Trades Risk Configuration Fixer")
        print("="*60)
        
        # Find or create config
        config_path = self.find_config_file()
        
        if config_path:
            if not self.load_config(config_path):
                print("❌ Failed to load configuration. Please check the file format.")
                return False
            
            # Create backup before making changes
            if not self.backup_config():
                print("⚠️  Warning: Could not create backup. Proceeding anyway...")
        else:
            if not self.create_default_config():
                print("❌ Failed to create default configuration.")
                return False
        
        # Fix risk configuration
        self.fix_risk_config()
        
        if self.changes_made:
            if not self.save_config():
                print("❌ Failed to save configuration changes.")
                return False
            
            print("\n✅ Configuration fixed successfully!")
            print("\nNext steps:")
            print("1. Review the changes in the config file")
            print("2. Update any API keys or other sensitive information")
            print("3. Run the validation again: python validate_production.py")
        else:
            print("\n✅ No changes needed. Your configuration looks good!")
        
        return True

def main():
    fixer = ConfigFixer()
    if fixer.run():
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
