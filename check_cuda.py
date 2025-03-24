#!/usr/bin/env python
"""
CUDA Environment Checker

This script checks your CUDA environment and compatibility with CuPy.
"""

import os
import sys
import subprocess
import platform

def run_command(cmd):
    """Run a command and return its output."""
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"Error ({e.returncode}): {e.output.decode('utf-8')}"

def check_system():
    """Check system information."""
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")

def check_cuda():
    """Check CUDA installation."""
    print("\n=== CUDA Information ===")
    
    # Check NVIDIA driver
    nvidia_smi = run_command("nvidia-smi")
    if "NVIDIA-SMI" in nvidia_smi:
        print("NVIDIA driver installed:")
        for line in nvidia_smi.split("\n")[:3]:
            print(f"  {line}")
    else:
        print("NVIDIA driver not found or not working properly")
    
    # Check CUDA version
    nvcc_version = run_command("nvcc --version")
    if "release" in nvcc_version:
        for line in nvcc_version.split("\n"):
            if "release" in line:
                print(f"NVCC: {line.strip()}")
    else:
        print("NVCC not found. CUDA toolkit might not be installed properly")
    
    # Check CUDA paths
    print("\nCUDA Paths:")
    cuda_path = os.environ.get("CUDA_PATH", "Not set")
    print(f"  CUDA_PATH: {cuda_path}")
    
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "Not set")
    print(f"  LD_LIBRARY_PATH: {ld_library_path}")

def check_python_packages():
    """Check Python packages for CUDA compatibility."""
    print("\n=== Python CUDA Packages ===")
    
    # Try importing CuPy
    try:
        print("\nChecking CuPy:")
        import cupy
        print(f"  CuPy version: {cupy.__version__}")
        print(f"  CUDA version detected by CuPy: {cupy.cuda.runtime.runtimeGetVersion()}")
        
        # Get GPU info
        device = cupy.cuda.Device(0)
        print(f"  GPU: {device.attributes['name'].decode()}")
        print(f"  Global memory: {device.attributes['totalGlobalMem'] / (1024**3):.2f} GB")
        print(f"  Compute capability: {device.attributes['computeCapabilityMajor']}.{device.attributes['computeCapabilityMinor']}")
        
        # Run a simple test
        print("\nRunning CuPy test (matrix multiplication)...")
        a = cupy.random.random((1000, 1000))
        b = cupy.random.random((1000, 1000))
        c = cupy.dot(a, b)
        print("  CuPy test successful!")
        
    except ImportError:
        print("  CuPy not installed")
    except Exception as e:
        print(f"  Error testing CuPy: {str(e)}")
    
    # Try importing PyCUDA
    try:
        print("\nChecking PyCUDA:")
        import pycuda.driver as cuda
        import pycuda.autoinit
        print(f"  PyCUDA version: {pycuda.VERSION}")
        print(f"  CUDA driver version: {cuda.get_version()}")
        
        # Get device info
        device = cuda.Device(0)
        print(f"  GPU: {device.name()}")
        print(f"  Compute capability: {device.compute_capability()}")
        print(f"  Total memory: {device.total_memory() / (1024**3):.2f} GB")
        
    except ImportError:
        print("  PyCUDA not installed")
    except Exception as e:
        print(f"  Error testing PyCUDA: {str(e)}")

def main():
    """Main function."""
    print("=== CUDA Environment Check ===\n")
    
    check_system()
    check_cuda()
    check_python_packages()
    
    print("\nEnvironment check completed.")

if __name__ == "__main__":
    main()
