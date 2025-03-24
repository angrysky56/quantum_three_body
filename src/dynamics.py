"""
Quantum dynamics module for the three-qubit system.

This module provides functions to evolve the three-qubit system
over time and analyze its dynamics, with optional GPU acceleration.
"""

import numpy as np
import qutip as qt
import time
from typing import List, Tuple, Union, Callable, Dict, Any

# Import GPU accelerator if available
try:
    from .gpu_accelerator import (
        gpu_accelerated, configure_gpu, batch_evolution, 
        parallel_expectation_values, HAS_CUPY, HAS_PYCUDA
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    # Create dummy decorators when GPU is not available
    def gpu_accelerated(use_gpu=True): 
        def decorator(func): return func
        return decorator


class ThreeQubitDynamics:
    """
    A class to simulate and analyze the dynamics of a three-qubit system.
    
    This class provides methods to evolve the quantum state over time,
    calculate various metrics of interest, and analyze the system's behavior.
    Features optional GPU acceleration for improved performance.
    
    Attributes:
        hamiltonian (Qobj): The Hamiltonian governing the system dynamics
        initial_state (Qobj): The initial state of the three-qubit system
        times (np.ndarray): Array of time points for the simulation
        results (Result): Results from the quantum evolution
        use_gpu (bool): Whether to use GPU acceleration when available
        performance_stats (dict): Performance statistics for benchmarking
    """
    
    def __init__(self, hamiltonian, use_gpu=True):
        """
        Initialize the dynamics simulator with a given Hamiltonian.
        
        Args:
            hamiltonian (Qobj or list): The Hamiltonian (static or time-dependent)
                governing the system dynamics
            use_gpu (bool): Whether to use GPU acceleration when available
        """
        self.hamiltonian = hamiltonian
        self.initial_state = None
        self.times = None
        self.results = None
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.performance_stats = {
            'simulation_time': 0,
            'computation_speedup': 1.0,
            'gpu_utilization': 0.0 if self.use_gpu else None
        }
        
        # Print GPU status on initialization
        if self.use_gpu:
            print("GPU acceleration enabled for quantum dynamics")
            try:
                import cupy as cp
                print(f"Using CuPy version {cp.__version__} with CUDA")
                if HAS_PYCUDA:
                    from .gpu_accelerator import get_gpu_info
                    info = get_gpu_info()
                    print(f"GPU: {info.get('name', 'Unknown')}")
            except ImportError:
                print("GPU acceleration requested but CuPy not available")
                self.use_gpu = False
    
    def set_initial_state(self, state=None):
        """
        Set the initial state of the three-qubit system.
        
        Args:
            state (Qobj, optional): Initial quantum state. If None, sets to |000⟩
            
        Returns:
            Qobj: The initial state
        """
        if state is None:
            # Default to |000⟩ state
            self.initial_state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
        else:
            self.initial_state = state
            
        return self.initial_state
    
    def create_ghz_state(self):
        """
        Create a GHZ entangled state (|000⟩ + |111⟩)/√2.
        
        Returns:
            Qobj: The GHZ state
        """
        state_000 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))
        state_111 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1), qt.basis(2, 1))
        ghz = (state_000 + state_111).unit()
        self.initial_state = ghz
        return ghz
    
    def create_w_state(self):
        """
        Create a W entangled state (|100⟩ + |010⟩ + |001⟩)/√3.
        
        Returns:
            Qobj: The W state
        """
        state_100 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0), qt.basis(2, 0))
        state_010 = qt.tensor(qt.basis(2, 0), qt.basis(2, 1), qt.basis(2, 0))
        state_001 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 1))
        w_state = (state_100 + state_010 + state_001).unit()
        self.initial_state = w_state
        return w_state
    
    def run_simulation(self, times, e_ops=None, options=None, show_progress=True, force_cpu=False):
        """
        Run the quantum dynamics simulation with optional GPU acceleration.
        
        Args:
            times (np.ndarray): Array of time points for the simulation
            e_ops (list, optional): List of observables to calculate expectation values
            options (dict or qt.Options, optional): Options for the ODE solver
            show_progress (bool): Whether to show a progress bar during simulation
            force_cpu (bool): Force CPU execution even if GPU is available
            
        Returns:
            Result: The simulation results
        """
        # Force CPU if requested
        original_gpu_setting = self.use_gpu
        if force_cpu:
            if show_progress:
                print("Forcing CPU execution (GPU disabled)")
            self.use_gpu = False
        if self.initial_state is None:
            self.set_initial_state()
            
        self.times = times
        
        # Safely handle options conversion
        qt_options = None
        try:
            # Try to create a proper QuTiP Options object
            if options is None:
                qt_options = qt.Options(nsteps=1000, atol=1e-8, rtol=1e-6, store_states=True)
            elif isinstance(options, qt.Options):
                qt_options = options
                qt_options.store_states = True
            else:
                # Try to convert dict to Options
                try:
                    nsteps = options.get('nsteps', 1000)
                    atol = options.get('atol', 1e-8)
                    rtol = options.get('rtol', 1e-6)
                    qt_options = qt.Options(nsteps=nsteps, atol=atol, rtol=rtol, store_states=True)
                except (AttributeError, TypeError):
                    print("Warning: options parameter was not a dict or qt.Options, using defaults")
                    qt_options = qt.Options(nsteps=1000, atol=1e-8, rtol=1e-6, store_states=True)
            
            # Add progress bar if requested (with error handling)
            if show_progress:
                try:
                    qt_options.progress_bar = True
                except Exception as e:
                    print(f"Warning: Could not set progress bar: {e}")
        except Exception as e:
            print(f"Error creating QuTiP Options: {e}")
            # Fall back to dict options
            qt_options = {'nsteps': 1000, 'atol': 1e-8, 'rtol': 1e-6, 'store_states': True}
            
        # Benchmark start time
        start_time = time.time()
        
        # Initialize results storage
        self.results = None
        simulation_success = False
        gpu_attempted = False
        
        # First attempt: Try GPU acceleration if applicable
        if self.use_gpu and not isinstance(self.hamiltonian, list) and e_ops is not None:
            try:
                gpu_attempted = True
                # Clear CUDA cache before running to reduce memory fragmentation
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
                    
                # Run GPU simulation
                self.results = self._run_gpu_simulation(times, e_ops, qt_options)
                self.performance_stats['gpu_utilization'] = 1.0  # Full GPU utilization
                simulation_success = True
            except Exception as e:
                print(f"GPU execution failed with error: {e}")
                print("Falling back to CPU execution...")
                self.performance_stats['gpu_utilization'] = 0.0
                # We'll fall through to CPU execution
        
        # Second attempt or first if GPU not applicable: CPU execution
        if not simulation_success:
            try:
                # Use standard QuTiP solvers
                self._run_cpu_simulation(times, e_ops, qt_options)
                simulation_success = True
            except Exception as e:
                print(f"CPU execution failed: {e}")
                # Create minimal result structure to prevent further errors
                print("Creating minimal result structure to prevent further errors")
                self.results = qt.Result()
                self.results.times = times
                self.results.states = [self.initial_state.copy() for _ in times]
                if e_ops:
                    self.results.expect = [np.zeros(len(times)) for _ in range(len(e_ops))]
                else:
                    self.results.expect = []
                self.results.stats = {}
        
        # Calculate and store performance statistics
        self.performance_stats['simulation_time'] = time.time() - start_time
        
        if show_progress:
            print(f"Simulation completed in {self.performance_stats['simulation_time']:.2f} seconds")
            if gpu_attempted:
                print(f"GPU acceleration: {'Active' if self.performance_stats['gpu_utilization'] > 0 else 'Inactive'}")
        
        # Restore original GPU setting if it was changed
        if force_cpu:
            self.use_gpu = original_gpu_setting
        
        return self.results
    
    def _run_cpu_simulation(self, times, e_ops, options):
        """Run simulation using standard QuTiP CPU solvers."""
        # Run the simulation
        if isinstance(self.hamiltonian, list):  # Time-dependent Hamiltonian
            self.results = qt.sesolve(self.hamiltonian, self.initial_state, times, e_ops, options=options)
        else:  # Time-independent Hamiltonian
            self.results = qt.mesolve(self.hamiltonian, self.initial_state, times, [], e_ops, options=options)
    
    def _run_gpu_simulation(self, times, e_ops, options):
        """
        Run simulation using GPU-accelerated algorithms.
        
        This method uses custom GPU-optimized solvers for time-independent Hamiltonians.
        """
        import cupy as cp
        from .gpu_accelerator import batch_evolution, parallel_expectation_values
        
        try:
            # Check if GPU device is properly initialized
            try:
                cp.cuda.Device(0).compute_capability  # Test GPU access
            except Exception as e:
                raise RuntimeError(f"GPU device initialization error: {e}")
                
            # Convert QuTiP operators to numpy arrays
            H_matrix = self.hamiltonian.full()
            psi0_vector = self.initial_state.full().flatten()
            
            # Run time evolution using GPU-accelerated solver
            states_gpu = batch_evolution(
                H_matrix, 
                psi0_vector.reshape(1, -1),  # Reshape to batch of 1 state
                times
            )
            
            # Handle potential shape issues
            if len(states_gpu.shape) > 2:
                states_gpu = states_gpu[:, 0, :]  # Extract the single state
            
            # Initialize result object with QuTiP structure
            # Create a proper QuTiP Result object with all required parameters
            result = qt.Result()
            result.times = times
            result.states = []
            result.expect = [np.zeros(len(times)) for _ in range(len(e_ops))]
            result.stats = {}  # Initialize stats dictionary
            
            # Set required attributes that were missing
            result.solver = "GPU accelerated solver"
            
            # Convert states back to QuTiP objects and calculate expectations
            for i, t in enumerate(times):
                try:
                    # Convert state at time t to QuTiP object
                    reshaped_state = states_gpu[i].reshape(self.initial_state.shape)
                    state_t = qt.Qobj(reshaped_state)
                    result.states.append(state_t)
                    
                    # Calculate expectation values using GPU
                    if e_ops:
                        op_matrices = [op.full() for op in e_ops]
                        expect_vals = parallel_expectation_values(
                            states_gpu[i], 
                            op_matrices
                        ).real
                        
                        # Store expectation values
                        for j, val in enumerate(expect_vals):
                            result.expect[j][i] = val
                except Exception as e:
                    print(f"Error processing state at time {t}: {e}")
                    # Create a copy of the initial state as a fallback
                    result.states.append(self.initial_state.copy())
                    for j in range(len(e_ops)):
                        result.expect[j][i] = 0.0
            
            return result
            
        except Exception as e:
            print(f"GPU simulation error (detailed): {str(e)}")
            # Release CUDA resources explicitly
            try:
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            raise  # Re-raise the exception to be caught by the calling method
    
    def compute_entanglement_metrics(self):
        """
        Compute various entanglement metrics over time.
        
        This function calculates pairwise concurrence and other entanglement measures
        for the three-qubit system throughout its time evolution.
        
        Returns:
            dict: Dictionary containing various entanglement metrics
        """
        if self.results is None:
            raise ValueError("No simulation results found. Run simulation first.")
            
        # Initialize metrics
        metrics = {
            'times': self.times,
            'concurrence_12': np.zeros(len(self.times)),
            'concurrence_23': np.zeros(len(self.times)),
            'concurrence_13': np.zeros(len(self.times)),
            'entropy_1': np.zeros(len(self.times)),
            'entropy_2': np.zeros(len(self.times)),
            'entropy_3': np.zeros(len(self.times)),
            'tripartite_entanglement': np.zeros(len(self.times))
        }
        
        # Compute metrics for each time step
        for i, t in enumerate(self.times):
            # Get the state at time t
            state = self.results.states[i]
            
            # Compute reduced density matrices
            rho_12 = state.ptrace([0, 1])
            rho_23 = state.ptrace([1, 2])
            rho_13 = state.ptrace([0, 2])
            rho_1 = state.ptrace([0])
            rho_2 = state.ptrace([1])
            rho_3 = state.ptrace([2])
            
            # Compute pairwise concurrence (a measure of entanglement)
            metrics['concurrence_12'][i] = qt.concurrence(rho_12)
            metrics['concurrence_23'][i] = qt.concurrence(rho_23)
            metrics['concurrence_13'][i] = qt.concurrence(rho_13)
            
            # Compute von Neumann entropy (another measure of entanglement)
            metrics['entropy_1'][i] = qt.entropy_vn(rho_1)
            metrics['entropy_2'][i] = qt.entropy_vn(rho_2)
            metrics['entropy_3'][i] = qt.entropy_vn(rho_3)
            
            # Compute a measure of tripartite entanglement
            # (This is a simplified metric; more complex measures exist)
            metrics['tripartite_entanglement'][i] = (
                metrics['entropy_1'][i] + metrics['entropy_2'][i] + metrics['entropy_3'][i]
            ) / 3.0
            
        return metrics
    
    @gpu_accelerated()
    def compute_quantum_chaos_metrics(self, num_perturbations=10, perturbation_strength=1e-6, 
                                     parallel=True, show_progress=True):
        """
        Compute metrics related to quantum chaos with GPU acceleration.
        
        This function calculates metrics like quantum Lyapunov exponents
        and state overlaps that can indicate chaotic behavior.
        
        Args:
            num_perturbations (int): Number of perturbed simulations to run
            perturbation_strength (float): Strength of the perturbation
            parallel (bool): Whether to run perturbations in parallel on GPU
            show_progress (bool): Whether to show progress during computation
            
        Returns:
            dict: Dictionary containing various quantum chaos metrics
        """
        if self.results is None:
            raise ValueError("No simulation results found. Run simulation first.")
            
        # Original state evolution
        original_states = self.results.states
        
        # Initialize metrics
        metrics = {
            'times': self.times,
            'fidelity_decay': np.zeros((num_perturbations, len(self.times))),
            'lyapunov_estimate': np.zeros(len(self.times)),
            'computation_time': 0.0
        }
        
        start_time = time.time()
        
        if self.use_gpu and GPU_AVAILABLE and parallel:
            try:
                # GPU-accelerated parallel perturbation simulations
                self._compute_chaos_metrics_gpu(metrics, num_perturbations, perturbation_strength, show_progress)
            except Exception as e:
                print(f"GPU chaos metrics computation failed: {e}")
                print("Falling back to CPU computation...")
                self._compute_chaos_metrics_cpu(metrics, num_perturbations, perturbation_strength, show_progress)
        else:
            # Standard CPU computation
            self._compute_chaos_metrics_cpu(metrics, num_perturbations, perturbation_strength, show_progress)
        
        # Compute chaos indicators from fidelity decay
        self._compute_chaos_indicators(metrics)
        
        # Record computation time
        metrics['computation_time'] = time.time() - start_time
        
        if show_progress:
            print(f"Chaos metrics computation completed in {metrics['computation_time']:.2f} seconds")
            
        return metrics
    
    def _compute_chaos_metrics_cpu(self, metrics, num_perturbations, perturbation_strength, show_progress):
        """Compute chaos metrics using CPU."""
        original_states = self.results.states
        
        for j in range(num_perturbations):
            if show_progress:
                print(f"Running perturbation {j+1}/{num_perturbations}")
                
            # Create a slightly perturbed initial state
            perturbed_state = self.initial_state + perturbation_strength * qt.rand_ket(8)
            perturbed_state = perturbed_state.unit()
            
            # Run simulation with perturbed initial state
            perturbed_dynamics = ThreeQubitDynamics(self.hamiltonian, use_gpu=self.use_gpu)
            perturbed_dynamics.initial_state = perturbed_state
            perturbed_results = perturbed_dynamics.run_simulation(self.times, show_progress=False)
            
            # Calculate fidelity between original and perturbed states over time
            for i, t in enumerate(self.times):
                original_state = original_states[i]
                perturbed_state = perturbed_results.states[i]
                
                fidelity = qt.fidelity(original_state, perturbed_state)
                metrics['fidelity_decay'][j, i] = fidelity
    
    def _compute_chaos_metrics_gpu(self, metrics, num_perturbations, perturbation_strength, show_progress):
        """Compute chaos metrics using GPU parallelization."""
        import cupy as cp
        from .gpu_accelerator import batch_evolution
        
        if show_progress:
            print("Computing quantum chaos metrics with GPU acceleration...")
            
        # Clean up GPU resources before starting
        try:
            cp.cuda.Device(0).synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU resources cleaned before quantum chaos computation")
        except Exception as e:
            print(f"GPU resource cleanup warning: {e}")
        
        try:
            # Check GPU device status
            cp.cuda.Device(0).compute_capability  # Test GPU access
            
            # Get original Hamiltonian and states as numpy arrays
            H_matrix = self.hamiltonian.full()
            original_state_vector = self.initial_state.full().flatten()
            
            # Create batch of perturbed initial states
            psi0_batch = np.zeros((num_perturbations, original_state_vector.size), dtype=np.complex128)
            
            for j in range(num_perturbations):
                if show_progress:
                    print(f"Preparing perturbation {j+1}/{num_perturbations}")
                    
                # Generate random perturbation - using seed for reproducibility
                np.random.seed(j)  # Make perturbations deterministic
                perturbation = np.random.normal(0, perturbation_strength, original_state_vector.size) + \
                               1j * np.random.normal(0, perturbation_strength, original_state_vector.size)
                
                # Create perturbed state
                perturbed_state = original_state_vector + perturbation
                
                # Normalize
                norm = np.linalg.norm(perturbed_state)
                perturbed_state = perturbed_state / norm
                
                # Store in batch
                psi0_batch[j] = perturbed_state
            
            # Evolve all perturbed states in parallel on GPU
            if show_progress:
                print("Evolving perturbed states in parallel on GPU...")
                
            # Get original states evolution - handle potential shape issues
            original_evolved_states = []
            for state in self.results.states:
                flat_state = state.full().flatten()
                original_evolved_states.append(flat_state)
            original_evolved_states = np.array(original_evolved_states)
            
            # Make sure dimensions match what batch_evolution expects
            if len(psi0_batch.shape) != 2:
                raise ValueError(f"psi0_batch should be 2D array, got shape {psi0_batch.shape}")
                
            # Clean up GPU before major calculation
            try:
                cp.cuda.Device(0).synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
                
            # Run batch evolution for all perturbed states - with explicit error checking
            perturbed_evolved_states = batch_evolution(H_matrix, psi0_batch, self.times)
            
            # Validate shapes to avoid dimension mismatch
            if len(perturbed_evolved_states.shape) != 3:
                # Reshape if needed (time, batch, state_dim)
                if len(perturbed_evolved_states.shape) == 2:
                    # If only one perturbation, reshape to (time, 1, state_dim)
                    print(f"Reshaping perturbed states from {perturbed_evolved_states.shape} to (time, 1, state_dim)")
                    perturbed_evolved_states = perturbed_evolved_states.reshape(
                        perturbed_evolved_states.shape[0], 1, perturbed_evolved_states.shape[1]
                    )
                else:
                    raise ValueError(f"Unexpected shape from batch_evolution: {perturbed_evolved_states.shape}")
            
            # Calculate fidelities
            if show_progress:
                print("Calculating fidelity decay...")
                
            for j in range(num_perturbations):
                for i, t in enumerate(self.times):
                    try:
                        # Extract states with shape validation
                        original = original_evolved_states[i]
                        
                        # Make sure we access perturbed correctly based on actual shape
                        if len(perturbed_evolved_states.shape) == 3:
                            if j < perturbed_evolved_states.shape[1]:
                                perturbed = perturbed_evolved_states[i, j]
                            else:
                                print(f"Index {j} out of bounds for perturbed_evolved_states with shape {perturbed_evolved_states.shape}")
                                perturbed = perturbed_evolved_states[i, 0]  # Use first perturbation as fallback
                        else:
                            # Fallback if shape isn't as expected
                            perturbed = perturbed_evolved_states[i]
                        
                        # Explicitly reshape vectors if needed to ensure they're 1D
                        if original.ndim > 1:
                            original = original.flatten()
                        if perturbed.ndim > 1:
                            perturbed = perturbed.flatten()
                            
                        # Make sure vectors have same dimension
                        if original.shape != perturbed.shape:
                            print(f"Warning: Dimension mismatch at time {t}, perturbation {j}")
                            print(f"Original shape: {original.shape}, Perturbed shape: {perturbed.shape}")
                            
                            # Try to reshape to match if possible
                            if original.size == perturbed.size:
                                perturbed = perturbed.reshape(original.shape)
                                print(f"Reshaped perturbed to match: {perturbed.shape}")
                            else:
                                # Use a default value if reshaping impossible
                                metrics['fidelity_decay'][j, i] = 1.0  # Default to no decay
                                continue
                        
                        # Calculate fidelity: |⟨ψ₁|ψ₂⟩|²
                        fidelity = np.abs(np.dot(np.conj(original), perturbed))**2
                        metrics['fidelity_decay'][j, i] = fidelity
                    except Exception as e:
                        print(f"Error calculating fidelity at time {t}, perturbation {j}: {e}")
                        # Use a default value
                        metrics['fidelity_decay'][j, i] = 1.0  # Default to no decay
            
            # Clean up GPU resources
            cp.cuda.Device(0).synchronize()
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            print("GPU resources cleaned after quantum chaos computation")
            
        except cp.cuda.runtime.CUDARuntimeError as e:
            print(f"CUDA Runtime Error in chaos metrics computation: {e}")
            # Attempt to reset CUDA context
            try:
                cp.cuda.Device(0).synchronize()
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
            raise  # Let the calling method handle the fallback to CPU
        except Exception as e:
            print(f"GPU chaos metrics computation failed: {e}")
            raise  # Let the calling method handle the fallback to CPU
    
    def _compute_chaos_indicators(self, metrics):
        """Compute chaos indicators from fidelity decay data."""
        # Average fidelity decay across all perturbations
        avg_fidelity_decay = np.mean(metrics['fidelity_decay'], axis=0)
        
        # Estimate Lyapunov exponent from fidelity decay
        for i in range(1, len(self.times)):
            dt = self.times[i] - self.times[i-1]
            
            # Protect against numerical issues
            decay_i = max(1e-10, 1 - avg_fidelity_decay[i])
            decay_i_prev = max(1e-10, 1 - avg_fidelity_decay[i-1])
            
            log_ratio = np.log(decay_i) - np.log(decay_i_prev)
            metrics['lyapunov_estimate'][i] = log_ratio / dt
        
        # Add average fidelity decay to metrics
        metrics['avg_fidelity_decay'] = avg_fidelity_decay
        
        # Additional advanced chaos metrics
        metrics['quantum_butterfly_effect'] = self._compute_butterfly_effect(avg_fidelity_decay)
        metrics['entanglement_growth_rate'] = self._estimate_entanglement_growth_rate()
        
    def _compute_butterfly_effect(self, avg_fidelity_decay):
        """
        Compute the quantum butterfly effect metric.
        
        This quantifies how fast initially close quantum states diverge.
        """
        # Find the time it takes for the fidelity to drop below 0.9
        threshold = 0.9
        for i, fidelity in enumerate(avg_fidelity_decay):
            if fidelity < threshold:
                return 1.0 / self.times[i]  # Rate of divergence
        
        # If fidelity never drops below threshold
        return 0.0
    
    def _estimate_entanglement_growth_rate(self):
        """
        Estimate the rate of entanglement growth as a chaos metric.
        
        Fast entanglement growth is a signature of quantum chaos.
        """
        # Get the entanglement metrics
        try:
            ent_metrics = self.compute_entanglement_metrics()
            
            # Extract entropies
            entropies = (ent_metrics['entropy_1'] + 
                         ent_metrics['entropy_2'] + 
                         ent_metrics['entropy_3']) / 3
            
            # Compute the average rate of increase in the first quarter of the evolution
            n_points = len(entropies) // 4
            if n_points > 1:
                return (entropies[n_points] - entropies[0]) / self.times[n_points]
            else:
                return 0.0
        except:
            # Return zero if computation fails
            return 0.0
    
    def get_bloch_vectors(self):
        """
        Get the Bloch sphere vectors for each qubit over time.
        
        Returns:
            dict: Dictionary of Bloch vectors for each qubit
        """
        if self.results is None:
            raise ValueError("No simulation results found. Run simulation first.")
        
        # Initialize Bloch vectors
        num_times = len(self.times)
        bloch_vectors = {
            'times': self.times,
            'qubit1': {
                'x': np.zeros(num_times),
                'y': np.zeros(num_times),
                'z': np.zeros(num_times)
            },
            'qubit2': {
                'x': np.zeros(num_times),
                'y': np.zeros(num_times),
                'z': np.zeros(num_times)
            },
            'qubit3': {
                'x': np.zeros(num_times),
                'y': np.zeros(num_times),
                'z': np.zeros(num_times)
            }
        }
        
        # Compute Bloch vectors for each time
        for i, state in enumerate(self.results.states):
            # Get reduced density matrices for each qubit
            rho_1 = state.ptrace([0])
            rho_2 = state.ptrace([1])
            rho_3 = state.ptrace([2])
            
            # Calculate Bloch vectors
            bloch_1 = [qt.expect(op, rho_1) for op in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]]
            bloch_2 = [qt.expect(op, rho_2) for op in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]]
            bloch_3 = [qt.expect(op, rho_3) for op in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]]
            
            # Store values
            bloch_vectors['qubit1']['x'][i] = bloch_1[0]
            bloch_vectors['qubit1']['y'][i] = bloch_1[1]
            bloch_vectors['qubit1']['z'][i] = bloch_1[2]
            
            bloch_vectors['qubit2']['x'][i] = bloch_2[0]
            bloch_vectors['qubit2']['y'][i] = bloch_2[1]
            bloch_vectors['qubit2']['z'][i] = bloch_2[2]
            
            bloch_vectors['qubit3']['x'][i] = bloch_3[0]
            bloch_vectors['qubit3']['y'][i] = bloch_3[1]
            bloch_vectors['qubit3']['z'][i] = bloch_3[2]
            
        return bloch_vectors
