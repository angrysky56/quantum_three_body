#!/usr/bin/env python
"""
Quantum Three-Body Simulation Launcher

This script properly sets up the Python environment and launches
the simulation notebook with correct imports.
"""

import os
import sys
import platform
import subprocess
import importlib.util

def check_gpu_availability():
    """Check if NVIDIA GPU is available and configured."""
    try:
        # Try to import CuPy
        import cupy
        print(f"✓ CuPy version {cupy.__version__} found")
        
        # Get GPU info
        gpu_info = cupy.cuda.runtime.getDeviceProperties(0)
        print(f"✓ GPU: {gpu_info['name'].decode()}")
        print(f"✓ Compute Capability: {gpu_info['major']}.{gpu_info['minor']}")
        print(f"✓ Total Memory: {gpu_info['totalGlobalMem'] / 1e9:.2f} GB")
        return True
    except ImportError:
        print("✗ CuPy not found. GPU acceleration will not be available.")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed without disrupting existing ones."""
    # Core packages that shouldn't affect CuPy
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'qutip', 
        'jupyter', 'seaborn'
    ]
    
    # Remove ipython and pycuda from check list as they're handled separately
    # ipython is included with jupyter anyway
    
    # Use direct import check instead of importlib.util.find_spec
    # This avoids false negatives when packages are installed but not in the path
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        install = input("Would you like to install them now? (y/n): ")
        if install.lower() in ['y', 'yes']:
            # Install packages individually and carefully to preserve cupy
            for package in missing:
                print(f"Installing {package}...")
                try:
                    # First try installing without dependencies
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", package, "--no-deps"],
                        check=False  # Don't raise exception on error
                    )
                    
                    # Verify the installation worked
                    try:
                        __import__(package)
                        print(f"✓ {package} installed successfully")
                    except ImportError:
                        # If that didn't work, install with minimal dependencies
                        print(f"Installing {package} with dependencies...")
                        subprocess.run(
                            [sys.executable, "-m", "pip", "install", package, "--upgrade-strategy", "only-if-needed"],
                            check=False
                        )
                except Exception as e:
                    print(f"Warning: Could not install {package}: {e}")
        else:
            print("Please install the missing packages to continue.")
            return False
    
    # Check for cupy separately to preserve it
    try:
        import cupy
        print(f"✓ CuPy version {cupy.__version__} found")
    except ImportError:
        print("Note: CuPy not found. GPU acceleration will not be available.")
    
    return True

def setup_environment():
    """Set up the Python environment for the notebook."""
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Set environment variables for GPU configuration
    os.environ['QUANTUM_THREE_BODY_GPU'] = 'True'  # Enable GPU acceleration
    
    # Create initialization file for notebooks
    init_code = """
# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

# Import GPU configuration
try:
    from src.gpu_accelerator import configure_gpu
    configure_gpu(enable=True)  # Enable GPU acceleration
except ImportError:
    print("GPU acceleration not available")

# Common imports for notebooks
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML

# Initialize plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
"""
    
    # Write initialization to a file notebooks can import
    init_file = os.path.join(project_root, 'notebooks', 'notebook_init.py')
    os.makedirs(os.path.dirname(init_file), exist_ok=True)
    with open(init_file, 'w') as f:
        f.write(init_code)
    
    print(f"✓ Environment configured. Notebook initialization file created at {init_file}")
    return True

def launch_notebook():
    """Launch the Jupyter notebook."""
    notebook_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks')
    
    print("\n--- Launching Jupyter Notebook ---")
    print(f"Opening notebook directory: {notebook_dir}")
    print("You can now run the demo simulation notebook.")
    print("Make sure to import notebook_init at the beginning of your notebook:\n")
    print("    from notebook_init import *")
    print("    from src.simulation import SimulationPresets")
    print("    from src.visualization import *\n")
    
    # Launch Jupyter Notebook
    subprocess.Popen(["jupyter", "notebook"], cwd=notebook_dir)

def main():
    """Main function to set up and launch the simulation."""
    print("\n=== Quantum Three-Body Simulation Setup ===\n")
    
    # Check for quick launch argument
    quick_launch = "--quick" in sys.argv
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    
    # Check system information
    system_info = platform.platform()
    print(f"System: {system_info}")
    
    # Check GPU availability (but don't try to install anything)
    try:
        import cupy
        print(f"✓ CuPy version {cupy.__version__} found - GPU acceleration available")
    except ImportError:
        print("Note: CuPy not found. GPU acceleration will not be available.")
    except Exception as e:
        print(f"Warning when checking GPU: {e}")
    
    # Check dependencies (skip if quick launch)
    if not quick_launch:
        if not check_dependencies():
            proceed = input("Continue anyway? (y/n): ")
            if proceed.lower() not in ['y', 'yes']:
                return
    else:
        print("Quick launch mode: Skipping dependency checks")
    
    # Set up environment
    if not setup_environment():
        return
    
    # Launch notebook
    launch_notebook()
    
    # Print quick launch tip
    print("\nTIP: Next time, use 'python run_notebook.py --quick' to skip dependency checks.")

if __name__ == "__main__":
    main()
