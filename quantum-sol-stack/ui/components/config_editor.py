"""
Configuration Editor Component

This module provides a user interface for editing the QuantumSol Stack configuration.
"""
import os
import json
import yaml
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional

# Initialize logger
import logging
logger = logging.getLogger(__name__)

def load_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file"""
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif file_path.endswith('.json'):
                return json.load(f)
            else:
                raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
    except Exception as e:
        logger.error(f"Error loading config file {file_path}: {e}")
        return {}

def save_config(file_path: str, config: Dict[str, Any]) -> bool:
    """Save configuration to a YAML or JSON file"""
    try:
        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif file_path.endswith('.json'):
                json.dump(config, f, indent=2)
            else:
                raise ValueError("Unsupported file format. Use .yaml, .yml, or .json")
        return True
    except Exception as e:
        logger.error(f"Error saving config file {file_path}: {e}")
        return False

def render_config_editor():
    """Render the configuration editor interface"""
    st.title("⚙️ Configuration Editor")
    
    # Configuration file selection
    config_dir = Path("../config")
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.json"))
    
    if not config_files:
        st.warning("No configuration files found in the config directory.")
        return
    
    selected_file = st.selectbox(
        "Select a configuration file",
        options=config_files,
        format_func=lambda x: x.name
    )
    
    # Load the selected config file
    config = load_config(selected_file)
    
    if not config:
        st.error("Failed to load configuration file.")
        return
    
    # Display config in an editable format
    st.subheader(f"Editing: {selected_file.name}")
    
    # Create a form for editing the config
    with st.form("config_form"):
        # Display the config as editable JSON
        edited_config = st.text_area(
            "Edit configuration (YAML format)",
            value=yaml.dump(config, default_flow_style=False, sort_keys=False),
            height=400
        )
        
        # Form actions
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            save_btn = st.form_submit_button("💾 Save Changes")
        
        with col2:
            reset_btn = st.form_submit_button("🔄 Reset to Defaults")
        
        with col3:
            st.write("")
        
        if save_btn:
            try:
                # Parse the edited YAML
                new_config = yaml.safe_load(edited_config)
                
                # Save the updated config
                if save_config(selected_file, new_config):
                    st.success("Configuration saved successfully!")
                else:
                    st.error("Failed to save configuration.")
            except Exception as e:
                st.error(f"Invalid YAML: {e}")
        
        if reset_btn:
            # Reload the original config
            config = load_config(selected_file)
            st.rerun()
    
    # Configuration validation
    st.subheader("Configuration Validation")
    
    if st.button("🔍 Validate Configuration"):
        with st.spinner("Validating configuration..."):
            # Add your validation logic here
            is_valid = True
            issues = []
            
            # Example validation (customize based on your needs)
            required_sections = ["api", "trading", "risk_management"]
            for section in required_sections:
                if section not in config:
                    is_valid = False
                    issues.append(f"Missing required section: {section}")
            
            if is_valid:
                st.success("Configuration is valid!")
            else:
                st.error("Configuration validation failed:")
                for issue in issues:
                    st.error(f"- {issue}")
    
    # Configuration backup/restore
    st.subheader("Backup & Restore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Backup Configuration"):
            backup_dir = Path("../config/backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = st.session_state.get("timestamp", "")
            backup_file = backup_dir / f"{selected_file.stem}_backup_{timestamp}{selected_file.suffix}"
            
            if save_config(backup_file, config):
                st.success(f"Backup created: {backup_file.name}")
            else:
                st.error("Failed to create backup.")
    
    with col2:
        backup_files = list(Path("../config/backups").glob(f"{selected_file.stem}_backup_*"))
        
        if backup_files:
            selected_backup = st.selectbox(
                "Select a backup to restore",
                options=backup_files,
                format_func=lambda x: x.name
            )
            
            if st.button("🔄 Restore Backup"):
                backup_config = load_config(selected_backup)
                if backup_config and save_config(selected_file, backup_config):
                    st.success("Configuration restored successfully!")
                    st.rerun()
                else:
                    st.error("Failed to restore configuration.")
        else:
            st.warning("No backups available.")

# Example usage
if __name__ == "__main__":
    render_config_editor()
