#!/usr/bin/env python
"""
Quick Start Launcher for Quantum Three-Body Simulation

This is a minimal script that just launches the Jupyter notebook
without any dependency checks or installation attempts.
"""

import os
import sys
import subprocess

def check_jupyter_port(port):
    """Check if a port is in use."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result == 0  # True if port is in use

def find_free_port(start_port=8888, max_attempts=20):
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not check_jupyter_port(port):
            return port
    return None  # No free ports found

def kill_jupyter_on_port(port):
    """Kill process using the specified port."""
    try:
        # For Linux/Mac
        if sys.platform != 'win32':
            cmd = f"lsof -ti:{port} | xargs kill -9"
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            # For Windows
            cmd = f"FOR /F \"tokens=5\" %P IN ('netstat -ano ^| findstr :{port}') DO taskkill /F /PID %P"
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Killed process on port {port}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to kill process on port {port}: {e}")
        return False

def main():
    """Launch the Jupyter notebook with minimal setup."""
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Set environment variables
    os.environ['QUANTUM_THREE_BODY_GPU'] = 'True'  # Enable GPU acceleration
    
    # Check if ports are in use
    default_port = 8888
    ports_to_check = [8888, 8889, 8890]
    blocked_ports = [port for port in ports_to_check if check_jupyter_port(port)]
    
    if blocked_ports:
        print(f"⚠️ Found Jupyter running on ports: {', '.join(str(p) for p in blocked_ports)}")
        cleanup = input("Clean up these ports before starting? (y/n): ")
        if cleanup.lower() in ['y', 'yes']:
            for port in blocked_ports:
                kill_jupyter_on_port(port)
    
    # Find a free port
    free_port = find_free_port(default_port)
    if free_port is None:
        print("❌ No free ports found. Please close existing Jupyter instances.")
        print("   Run: python cleanup_jupyter.py")
        return
    
    # Set custom port option
    port_option = f"--port={free_port}"
    print(f"🚀 Using port {free_port} for Jupyter")
    
    # Launch the notebook
    notebook_dir = os.path.join(project_root, 'notebooks')
    print(f"📓 Starting Jupyter notebook in: {notebook_dir}")
    
    # Launch with specific port and notebook directory
    subprocess.run(["jupyter", "notebook", port_option, "--no-browser"], cwd=notebook_dir)

if __name__ == "__main__":
    main()
