#!/usr/bin/env python
"""
Jupyter Port Cleanup Utility

This script identifies and terminates orphaned Jupyter notebook processes
that might be holding ports open.
"""

import os
import sys
import signal
import subprocess
import time
import json

def find_jupyter_processes():
    """Find all running Jupyter notebook processes and their ports."""
    processes = []
    
    try:
        # Get list of all running processes
        if sys.platform == 'win32':
            output = subprocess.check_output(["tasklist", "/FO", "CSV"]).decode('utf-8')
            # Windows processing - more complex to parse
            jupyter_pids = []
            for line in output.split('\n'):
                if 'jupyter' in line.lower() or 'python' in line.lower():
                    parts = line.strip('"').split('","')
                    if len(parts) >= 2:
                        jupyter_pids.append(parts[1])
        else:
            # Unix-like systems
            output = subprocess.check_output(["ps", "aux"]).decode('utf-8')
            jupyter_pids = []
            for line in output.split('\n'):
                if 'jupyter' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        jupyter_pids.append(parts[1])
        
        # For each process, get the port information if possible
        for pid in jupyter_pids:
            if sys.platform == 'win32':
                try:
                    # Windows netstat
                    netstat = subprocess.check_output(["netstat", "-ano"]).decode('utf-8')
                    for line in netstat.split('\n'):
                        if pid in line and ('127.0.0.1:8888' in line or '127.0.0.1:8889' in line or '127.0.0.1:8890' in line):
                            port = line.split(':')[1].split(' ')[0]
                            processes.append({'pid': pid, 'port': port})
                except Exception as e:
                    processes.append({'pid': pid, 'port': 'unknown'})
            else:
                try:
                    # Unix-like lsof
                    lsof = subprocess.check_output(["lsof", "-i", "-P", "-n"]).decode('utf-8')
                    for line in lsof.split('\n'):
                        if pid in line and ('LISTEN' in line):
                            parts = line.split()
                            for part in parts:
                                if ':' in part:
                                    port = part.split(':')[-1]
                                    if port.isdigit():
                                        processes.append({'pid': pid, 'port': port})
                except:
                    # Try netstat as alternative
                    try:
                        netstat = subprocess.check_output(["netstat", "-tunlp"]).decode('utf-8')
                        for line in netstat.split('\n'):
                            if pid in line:
                                parts = line.split()
                                for part in parts:
                                    if ':' in part:
                                        port = part.split(':')[-1]
                                        if port.isdigit():
                                            processes.append({'pid': pid, 'port': port})
                    except:
                        processes.append({'pid': pid, 'port': 'unknown'})
    
    except Exception as e:
        print(f"Error finding processes: {e}")
    
    return processes

def find_jupyter_servers():
    """Find running Jupyter servers using the 'jupyter server list' command."""
    servers = []
    try:
        output = subprocess.check_output(["jupyter", "server", "list", "--json"]).decode('utf-8')
        # Each line is a separate JSON object
        for line in output.strip().split('\n'):
            if line.strip():
                try:
                    server_info = json.loads(line)
                    servers.append(server_info)
                except json.JSONDecodeError:
                    # Skip lines that aren't valid JSON
                    pass
    except Exception as e:
        print(f"Error listing Jupyter servers: {e}")
        # Fall back to regular process finding
        pass
    
    return servers

def kill_process(pid):
    """Kill a process by its PID."""
    try:
        if sys.platform == 'win32':
            subprocess.check_call(["taskkill", "/F", "/PID", str(pid)])
        else:
            os.kill(int(pid), signal.SIGTERM)
        return True
    except Exception as e:
        print(f"Error killing process {pid}: {e}")
        return False

def stop_jupyter_server(server_info):
    """Stop a Jupyter server using its token and URL."""
    try:
        url = server_info.get('url', '')
        token = server_info.get('token', '')
        if url and token:
            subprocess.check_call([
                "jupyter", "server", "stop", url, "--token=" + token
            ])
            return True
    except Exception as e:
        print(f"Error stopping Jupyter server: {e}")
    return False

def cleanup_jupyter_ports():
    """Main function to clean up Jupyter ports."""
    print("Checking for Jupyter processes holding ports...")
    
    # First try to use the jupyter server list command
    servers = find_jupyter_servers()
    if servers:
        print(f"Found {len(servers)} Jupyter servers.")
        for server in servers:
            url = server.get('url', '')
            port = url.split(':')[-1].split('/')[0] if ':' in url else 'unknown'
            print(f"Found Jupyter server on port {port}: {url}")
            
            # Try to stop the server gracefully
            if stop_jupyter_server(server):
                print(f"Successfully stopped Jupyter server on port {port}")
            else:
                print(f"Failed to stop Jupyter server gracefully, will try force killing")
    
    # Find and kill running Jupyter processes
    processes = find_jupyter_processes()
    if processes:
        print(f"Found {len(processes)} Jupyter processes:")
        for proc in processes:
            print(f"  PID: {proc['pid']}, Port: {proc['port']}")
            
            confirm = input(f"Kill process {proc['pid']} on port {proc['port']}? (y/n): ")
            if confirm.lower() in ['y', 'yes']:
                if kill_process(proc['pid']):
                    print(f"Successfully killed process {proc['pid']}")
                else:
                    print(f"Failed to kill process {proc['pid']}")
    
    # Check common Jupyter ports
    common_ports = [8888, 8889, 8890, 8891, 8892]
    for port in common_ports:
        try:
            # Try to connect to the port to see if it's in use
            if sys.platform == 'win32':
                # Windows approach using netstat
                output = subprocess.check_output(["netstat", "-ano", f"| findstr :{port}"]).decode('utf-8')
                if output.strip():
                    print(f"Port {port} is still in use by an unknown process")
            else:
                # Unix approach using lsof
                output = subprocess.check_output(["lsof", "-i", f":{port}"]).decode('utf-8')
                if output.strip():
                    print(f"Port {port} is still in use by an unknown process")
        except:
            # If the command fails, the port is likely free
            pass
    
    # Final verification
    print("\nVerifying cleanup...")
    remaining = find_jupyter_processes()
    if remaining:
        print(f"There are still {len(remaining)} Jupyter processes running.")
        print("You might need to manually kill these processes or restart your computer.")
    else:
        print("All Jupyter processes have been terminated.")
        print("You should now be able to start a new Jupyter notebook server.")

if __name__ == "__main__":
    try:
        cleanup_jupyter_ports()
    except KeyboardInterrupt:
        print("\nCleanup interrupted by user.")
    except Exception as e:
        print(f"An error occurred during cleanup: {e}")
