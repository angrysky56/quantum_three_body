#!/usr/bin/env python
"""
Fix Jupyter Theme - Direct Solution

This script installs and applies a dark theme to Jupyter Notebook.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import shutil

def get_jupyter_config_dir():
    """Get Jupyter config directory."""
    try:
        # Try using jupyter command
        result = subprocess.run(
            ["jupyter", "--config-dir"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        # Default paths if command fails
        home = Path.home()
        if sys.platform == 'win32':
            return os.path.join(home, 'AppData', 'Roaming', 'jupyter')
        elif sys.platform == 'darwin':
            return os.path.join(home, 'Library', 'Jupyter')
        else:
            return os.path.join(home, '.jupyter')

def install_custom_css():
    """Install custom CSS directly to Jupyter's custom folder"""
    jupyter_dir = get_jupyter_config_dir()
    custom_dir = os.path.join(jupyter_dir, 'custom')
    
    # Create custom directory if it doesn't exist
    os.makedirs(custom_dir, exist_ok=True)
    
    # Create CSS content directly (no file reading required)
    css_content = """
/* Dark Theme for Jupyter Notebooks */

/* Main background */
body {
    background-color: #111 !important;
    color: #f8f8f2 !important;
}

/* Menu bar */
#header {
    background-color: #1e1e1e !important;
}
#menubar {
    background-color: #1e1e1e !important;
}
.navbar-default {
    background-color: #1e1e1e !important;
}

/* Notebook cells */
div.cell {
    background-color: #1e1e1e !important;
    border: 1px solid #333 !important;
}

/* Cell input area */
div.input_area {
    background-color: #282a36 !important;
    border: 1px solid #444 !important;
}

/* Code cells */
.CodeMirror {
    background-color: #282a36 !important;
    color: #f8f8f2 !important;
}

/* Markdown cells */
div.text_cell_render {
    background-color: #1e1e1e !important;
    color: #f8f8f2 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #8be9fd !important;
}

/* Output area */
div.output_area {
    background-color: #1e1e1e !important;
    color: #f8f8f2 !important;
}

/* Output text */
div.output_stdout {
    background-color: #1e1e1e !important;
}

/* Error output */
div.output_stderr {
    background-color: #661c1c !important;
}

/* Links */
a {
    color: #8be9fd !important;
}

/* Buttons */
.btn {
    background-color: #44475a !important;
    color: #f8f8f2 !important;
}

/* Selected cell */
div.cell.selected {
    border: 2px solid #bd93f9 !important;
}

/* Input/output prompts */
div.prompt {
    color: #6272a4 !important;
}

/* Tables */
.rendered_html table, .rendered_html th, .rendered_html tr, .rendered_html td {
    background-color: #282a36 !important;
    color: #f8f8f2 !important;
    border: 1px solid #44475a !important;
}

/* Tooltip */
.tooltip {
    background-color: #282a36 !important;
    color: #f8f8f2 !important;
    border: 1px solid #44475a !important;
}

/* Dropdown menu */
.dropdown-menu {
    background-color: #282a36 !important;
}
.dropdown-menu > li > a {
    color: #f8f8f2 !important;
}
.dropdown-menu > li > a:hover {
    background-color: #44475a !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
::-webkit-scrollbar-track {
    background: #282a36;
}
::-webkit-scrollbar-thumb {
    background: #44475a;
}
::-webkit-scrollbar-thumb:hover {
    background: #6272a4;
}
"""
    
    # Write CSS to custom.css
    css_path = os.path.join(custom_dir, 'custom.css')
    with open(css_path, 'w') as f:
        f.write(css_content)
    
    print(f"✅ Dark theme installed to: {css_path}")
    print("Theme will be applied when you restart Jupyter Notebook.")

def install_jupyterthemes():
    """Try to install and apply jupyterthemes as a reliable alternative"""
    print("Installing jupyterthemes package...")
    try:
        # Check if jupyterthemes is already installed
        subprocess.run(
            [sys.executable, "-m", "pip", "show", "jupyterthemes"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print("jupyterthemes is already installed.")
    except:
        # Install jupyterthemes
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "jupyterthemes"],
            check=True
        )
        print("jupyterthemes installed successfully.")
    
    # Apply a dark theme
    try:
        subprocess.run(
            ["jt", "-t", "monokai", "-fs", "11", "-nf", "roboto", "-tf", "roboto", "-cursw", "5"],
            check=True
        )
        print("✅ Monokai theme applied successfully.")
        return True
    except Exception as e:
        print(f"⚠️ Error applying theme with jupyterthemes: {e}")
        return False

def main():
    """Main function to fix Jupyter theme"""
    print("🎨 Fixing Jupyter Notebook Theme...\n")
    
    # Try method 1: jupyterthemes (most reliable)
    print("Method 1: Using jupyterthemes package")
    success = install_jupyterthemes()
    
    # Always also apply custom CSS as backup
    print("\nMethod 2: Installing custom CSS directly")
    install_custom_css()
    
    print("\n✅ Theme fix applied! Please restart Jupyter Notebook to see the changes.")
    print("   If the theme still isn't applied, try clearing your browser cache.")
    
    # Offer to restart Jupyter
    restart = input("\nWould you like to restart Jupyter Notebook now? (y/n): ")
    if restart.lower() in ['y', 'yes']:
        # Kill existing Jupyter processes
        if sys.platform == 'win32':
            subprocess.run('taskkill /F /IM jupyter-notebook.exe', shell=True)
        else:
            subprocess.run('pkill -f jupyter-notebook', shell=True)
        
        # Start with fixed notebook
        notebook_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'notebooks', 'fixed_notebook.ipynb')
        subprocess.Popen(["jupyter", "notebook", notebook_path])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please try installing a theme manually:")
        print("1. pip install jupyterthemes")
        print("2. jt -t monokai")
