#!/usr/bin/env python
"""
Fix Jupyter Notebook Dark Theme

This script installs a custom dark theme for Jupyter Notebook
to ensure consistent dark mode experience.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def get_jupyter_config_dir():
    """Get Jupyter config directory."""
    try:
        output = subprocess.check_output(["jupyter", "--config-dir"], 
                                       stderr=subprocess.STDOUT).decode('utf-8').strip()
        return output
    except subprocess.CalledProcessError:
        # Default locations if jupyter command fails
        home = Path.home()
        if sys.platform == 'win32':
            return os.path.join(home, 'AppData', 'Roaming', 'jupyter')
        elif sys.platform == 'darwin':
            return os.path.join(home, 'Library', 'Jupyter')
        else:
            return os.path.join(home, '.jupyter')

def install_custom_css():
    """Install custom CSS for Jupyter Notebook."""
    # Get paths
    jupyter_config_dir = get_jupyter_config_dir()
    custom_dir = os.path.join(jupyter_config_dir, 'custom')
    css_path = os.path.join(custom_dir, 'custom.css')
    
    # Create custom directory if it doesn't exist
    os.makedirs(custom_dir, exist_ok=True)
    
    # Copy our custom CSS
    source_css = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'notebooks', 'custom.css')
    
    if os.path.exists(source_css):
        with open(source_css, 'r') as src_file:
            css_content = src_file.read()
            
        with open(css_path, 'w') as dest_file:
            dest_file.write(css_content)
        
        print(f"✅ Custom dark theme installed to {css_path}")
    else:
        print("❌ Source CSS file not found!")

def configure_jupyter_dark_mode():
    """Configure Jupyter to use dark mode."""
    try:
        # Try to enable dark mode via nbextensions if installed
        subprocess.run([sys.executable, "-m", "pip", "install", "jupyterthemes"],
                      check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Apply a dark theme
        subprocess.run(["jt", "-t", "monokai", "-f", "fira", "-fs", "12"],
                      check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("✅ Applied dark Jupyter theme")
    except Exception as e:
        print(f"⚠️ Could not apply Jupyter theme: {e}")
    
    # Install our custom CSS regardless
    install_custom_css()

if __name__ == "__main__":
    print("📄 Installing Dark Theme for Jupyter Notebook...")
    configure_jupyter_dark_mode()
    print("\nRestart Jupyter Notebook for changes to take effect!")
