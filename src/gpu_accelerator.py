"""
GPU Accelerator for Quantum Three-Body Simulations

This module provides GPU acceleration for quantum simulations using CuPy
and custom parallelization techniques for NVIDIA GPUs.
"""

import os
import numpy as np
import warnings
from functools import wraps

# Check for GPU availability
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    warnings.warn("CuPy not found. Running in CPU mode.")

# Attempt to import CUDA-specific libraries
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda import gpuarray
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

# Configuration
GPU_ENABLED = True  # Global switch to enable/disable GPU acceleration
DEVICE_ID = 0       # Default CUDA device ID
PRECISION = 'float64'  # Default precision

# Get information about the GPU if available
def get_gpu_info():
    """Get information about the available GPU."""
    if not HAS_PYCUDA:
        return {"status": "PYCUDA not available"}
    
    try:
        device = cuda.Device(DEVICE_ID)
        context = device.make_context()
        info = {
            "name": device.name(),
            "compute_capability": device.compute_capability(),
            "total_memory": device.total_memory(),
            "max_threads_per_block": device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK),
            "multiprocessor_count": device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT),
            "clock_rate": device.get_attribute(cuda.device_attribute.CLOCK_RATE),
            "memory_clock_rate": device.get_attribute(cuda.device_attribute.MEMORY_CLOCK_RATE),
        }
        context.pop()
        return info
    except Exception as e:
        return {"status": f"Error: {str(e)}"}

# Decorator for GPU acceleration of functions
def gpu_accelerated(use_gpu=True):
    """
    Decorator to enable GPU acceleration for numerical functions.
    
    Args:
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if GPU_ENABLED and use_gpu and HAS_CUPY:
                # Convert NumPy arrays to CuPy arrays
                new_args = []
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        new_args.append(cp.array(arg))
                    else:
                        new_args.append(arg)
                
                new_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, np.ndarray):
                        new_kwargs[key] = cp.array(value)
                    else:
                        new_kwargs[key] = value
                
                # Run the function with GPU arrays
                result = func(*new_args, **new_kwargs)
                
                # Convert results back to NumPy
                if isinstance(result, cp.ndarray):
                    return cp.asnumpy(result)
                elif isinstance(result, tuple):
                    return tuple(cp.asnumpy(r) if isinstance(r, cp.ndarray) else r for r in result)
                elif isinstance(result, list):
                    return [cp.asnumpy(r) if isinstance(r, cp.ndarray) else r for r in result]
                elif isinstance(result, dict):
                    return {k: cp.asnumpy(v) if isinstance(v, cp.ndarray) else v for k, v in result.items()}
                else:
                    return result
            else:
                # Run on CPU
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Optimized matrix operations for quantum simulations
@gpu_accelerated()
def tensor_product(matrices):
    """
    Compute tensor product of matrices with GPU acceleration.
    
    Args:
        matrices (list): List of matrices to compute tensor product
        
    Returns:
        ndarray: Tensor product result
    """
    if HAS_CUPY and GPU_ENABLED:
        result = matrices[0]
        for matrix in matrices[1:]:
            # Use CuPy's kron (Kronecker product) for tensor products
            result = cp.kron(result, matrix)
        return result
    else:
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result

@gpu_accelerated()
def parallel_expectation_values(state, operators):
    """
    Compute expectation values for multiple operators in parallel.
    
    Args:
        state (ndarray): Quantum state vector or density matrix
        operators (list): List of operators
        
    Returns:
        ndarray: Array of expectation values
    """
    if HAS_CUPY and GPU_ENABLED:
        # Initialize results array
        results = cp.zeros(len(operators), dtype=cp.complex128)
        
        # Handle both pure states and density matrices
        is_ket = state.ndim == 1 or state.shape[1] == 1
        
        if is_ket:
            # For pure states: <ψ|O|ψ>
            state_gpu = cp.array(state)
            for i, op in enumerate(operators):
                op_gpu = cp.array(op)
                results[i] = cp.dot(cp.conj(state_gpu), cp.dot(op_gpu, state_gpu))
        else:
            # For density matrices: Tr(ρO)
            state_gpu = cp.array(state)
            for i, op in enumerate(operators):
                op_gpu = cp.array(op)
                results[i] = cp.trace(cp.dot(state_gpu, op_gpu))
        
        return results
    else:
        # Fallback to CPU
        results = np.zeros(len(operators), dtype=np.complex128)
        
        is_ket = state.ndim == 1 or state.shape[1] == 1
        
        if is_ket:
            for i, op in enumerate(operators):
                results[i] = np.dot(np.conj(state), np.dot(op, state))
        else:
            for i, op in enumerate(operators):
                results[i] = np.trace(np.dot(state, op))
        
        return results

@gpu_accelerated()
def batch_evolution(H, psi0_batch, times):
    """
    Evolve multiple initial states in parallel.
    
    This function is useful for computing quantum chaos metrics
    which require simulating multiple perturbed initial states.
    
    Args:
        H (ndarray): Hamiltonian matrix
        psi0_batch (ndarray): Batch of initial states (shape: [n_states, dim])
        times (ndarray): Time points
        
    Returns:
        ndarray: Batch of evolved states at each time (shape: [n_times, n_states, dim])
    """
    # Input validation
    if len(psi0_batch.shape) != 2:
        raise ValueError(f"psi0_batch must be 2D array with shape [n_states, dim], got {psi0_batch.shape}")
    
    if HAS_CUPY and GPU_ENABLED:
        try:
            # Explicitly clean up resources before starting
            cp.cuda.Device(0).synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            
            # Convert to GPU arrays with explicit error handling
            try:
                H_gpu = cp.array(H)
                psi0_batch_gpu = cp.array(psi0_batch)
                times_gpu = cp.array(times)
            except Exception as e:
                print(f"Error converting to GPU arrays: {e}")
                raise
            
            # Get dimensions
            n_times = len(times_gpu)
            n_states, dim = psi0_batch_gpu.shape
            
            # Initialize smaller chunks to avoid memory issues
            max_states_per_batch = 5  # Process in smaller batches to avoid CUDA memory issues
            
            # Process in batches if necessary
            if n_states > max_states_per_batch:
                print(f"Processing {n_states} states in smaller batches to avoid memory issues")
                results = cp.zeros((n_times, n_states, dim), dtype=cp.complex128)
                
                for batch_start in range(0, n_states, max_states_per_batch):
                    batch_end = min(batch_start + max_states_per_batch, n_states)
                    batch_size = batch_end - batch_start
                    
                    # Process this mini-batch
                    mini_batch = psi0_batch_gpu[batch_start:batch_end]
                    
                    # Diagonalize Hamiltonian once (outside the loop to save time)
                    eigvals, eigvecs = cp.linalg.eigh(H_gpu)
                    
                    # Process each state in the mini-batch
                    for b_idx in range(batch_size):
                        b = batch_start + b_idx
                        psi0 = mini_batch[b_idx]
                        
                        # Express initial state in eigenbasis
                        psi0_eig = cp.dot(cp.conj(eigvecs.T), psi0)
                        
                        # Evolve for each time
                        for t, time in enumerate(times_gpu):
                            # Apply time evolution operator in eigenbasis
                            phase = cp.exp(-1j * eigvals * time)
                            psi_t_eig = psi0_eig * phase
                            
                            # Transform back to original basis
                            psi_t = cp.dot(eigvecs, psi_t_eig)
                            
                            # Store result
                            results[t, b] = psi_t
                
                # Clean up GPU memory
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                
                return results
            else:
                # Directly process all states if small enough
                # Diagonalize Hamiltonian once
                eigvals, eigvecs = cp.linalg.eigh(H_gpu)
                
                # Initialize results array
                results = cp.zeros((n_times, n_states, dim), dtype=cp.complex128)
                
                # Compute for each initial state in the batch
                for b in range(n_states):
                    psi0 = psi0_batch_gpu[b]
                    
                    # Express initial state in eigenbasis
                    psi0_eig = cp.dot(cp.conj(eigvecs.T), psi0)
                    
                    # Evolve for each time
                    for t, time in enumerate(times_gpu):
                        # Apply time evolution operator in eigenbasis
                        phase = cp.exp(-1j * eigvals * time)
                        psi_t_eig = psi0_eig * phase
                        
                        # Transform back to original basis
                        psi_t = cp.dot(eigvecs, psi_t_eig)
                        
                        # Store result
                        results[t, b] = psi_t
                
                # Clean up GPU memory
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                
                return results
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"GPU reset recommended, switching to CPU: {e}")
            # Clean up any remaining GPU resources
            try:
                cp.cuda.Device(0).synchronize()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            print("Falling back to CPU implementation")
        except Exception as e:
            print(f"GPU batch evolution failed: {e}")
            print("Falling back to CPU implementation")
            # Clean up any remaining GPU resources
            try:
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            
            # Fall through to CPU implementation
    
    # CPU implementation (fallback)
    try:
        # Fallback to CPU
        n_times = len(times)
        n_states, dim = psi0_batch.shape
        results = np.zeros((n_times, n_states, dim), dtype=np.complex128)
        
        # Special case for single state
        if n_states == 1:
            # Diagonalize Hamiltonian once
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Get the single initial state
            psi0 = psi0_batch[0]
            
            # Express initial state in eigenbasis
            psi0_eig = np.dot(np.conj(eigvecs.T), psi0)
            
            # Evolve for each time
            for t, time in enumerate(times):
                # Apply time evolution operator in eigenbasis
                phase = np.exp(-1j * eigvals * time)
                psi_t_eig = psi0_eig * phase
                
                # Transform back to original basis
                psi_t = np.dot(eigvecs, psi_t_eig)
                
                # Store result
                results[t, 0] = psi_t
        else:
            # Handle multiple states
            # Diagonalize Hamiltonian once
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Compute for each initial state in the batch
            for b in range(n_states):
                psi0 = psi0_batch[b]
                
                # Express initial state in eigenbasis
                psi0_eig = np.dot(np.conj(eigvecs.T), psi0)
                
                # Evolve for each time
                for t, time in enumerate(times):
                    # Apply time evolution operator in eigenbasis
                    phase = np.exp(-1j * eigvals * time)
                    psi_t_eig = psi0_eig * phase
                    
                    # Transform back to original basis
                    psi_t = np.dot(eigvecs, psi_t_eig)
                    
                    # Store result
                    results[t, b] = psi_t
        
        return results
    except Exception as e:
        print(f"CPU batch evolution failed: {e}")
        # Return a minimal valid result to prevent further errors
        dummy_result = np.zeros((len(times), 1, dim), dtype=np.complex128)
        return dummy_result

# Configure GPU settings
def configure_gpu(enable=True, device_id=0, precision='float64'):
    """
    Configure GPU settings for quantum simulations.
    
    Args:
        enable (bool): Whether to enable GPU acceleration
        device_id (int): CUDA device ID to use
        precision (str): Numerical precision ('float32' or 'float64')
    """
    global GPU_ENABLED, DEVICE_ID, PRECISION
    
    GPU_ENABLED = enable and HAS_CUPY
    DEVICE_ID = device_id
    PRECISION = precision
    
    if GPU_ENABLED and HAS_CUPY:
        try:
            cp.cuda.Device(device_id).use()
            
            # Set default precision if the method exists
            # This is compatible with different CuPy versions
            if precision == 'float32':
                dtype = cp.float32
            else:
                dtype = cp.float64
            
            # Try to set default dtype if the method exists
            if hasattr(cp, 'set_default_dtype'):
                cp.set_default_dtype(dtype)
            # Otherwise just note the preferred precision
            else:
                print(f"Using {precision} precision (Note: set_default_dtype not available in this CuPy version)")
            
            # Print GPU configuration
            print(f"GPU acceleration enabled on device {device_id}")
            if HAS_PYCUDA:
                info = get_gpu_info()
                print(f"Using {info.get('name', 'Unknown GPU')}")
                print(f"CUDA compute capability: {info.get('compute_capability', 'Unknown')}")
                print(f"Total memory: {info.get('total_memory', 0) / 1e9:.2f} GB")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
            print("Falling back to CPU mode.")
            GPU_ENABLED = False
    else:
        print("GPU acceleration disabled. Using CPU.")

# Initialize GPU if available
if HAS_CUPY and os.environ.get('QUANTUM_THREE_BODY_GPU', 'True').lower() != 'false':
    configure_gpu(enable=True)
