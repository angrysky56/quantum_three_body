"""
Simulation utility module for the three-qubit system.

This module provides high-level functions to set up and run common
simulation scenarios for the three-qubit system.
"""

import numpy as np
import qutip as qt
from typing import Dict, List, Tuple, Any, Optional

from .hamiltonian import ThreeQubitHamiltonian
from .dynamics import ThreeQubitDynamics


class SimulationPresets:
    """
    A class providing preset simulations for the three-qubit system.
    
    This class offers convenient methods to run common simulation scenarios,
    including various Hamiltonians, initial states, and analysis methods.
    """
    
    @staticmethod
    def run_ising_model(J_coupling=1.0, h_field=0.5, initial_state_type='ghz',
                        tmax=10.0, nsteps=1000, compute_metrics=True):
        """
        Run a simulation with an Ising model Hamiltonian.
        
        Args:
            J_coupling (float): Strength of Ising interaction
            h_field (float): Strength of transverse field
            initial_state_type (str): Type of initial state ('ghz', 'w', or 'product')
            tmax (float): Maximum simulation time
            nsteps (int): Number of time steps
            compute_metrics (bool): Whether to compute entanglement metrics
            
        Returns:
            dict: Results dictionary containing simulation results and metrics
        """
        # Create Hamiltonian
        hamiltonian_model = ThreeQubitHamiltonian()
        hamiltonian = hamiltonian_model.create_ising_hamiltonian(
            J_coupling=J_coupling, h_field=h_field
        )
        
        # Set up dynamics
        dynamics = ThreeQubitDynamics(hamiltonian)
        
        # Set initial state
        if initial_state_type == 'ghz':
            dynamics.create_ghz_state()
        elif initial_state_type == 'w':
            dynamics.create_w_state()
        elif initial_state_type == 'product':
            # Start in |000⟩ state
            dynamics.set_initial_state()
        else:
            raise ValueError("initial_state_type must be 'ghz', 'w', or 'product'")
        
        # Run simulation
        times = np.linspace(0, tmax, nsteps)
        
        # Define expectation operators
        sx_list = [
            qt.tensor(qt.sigmax(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmax(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmax())
        ]
        
        sy_list = [
            qt.tensor(qt.sigmay(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmay(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmay())
        ]
        
        sz_list = [
            qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmaz())
        ]
        
        # Run simulation with expectation value calculations
        e_ops = sx_list + sy_list + sz_list
        results = dynamics.run_simulation(times, e_ops=e_ops)
        
        # Compute metrics if requested
        metrics = {}
        if compute_metrics:
            metrics['entanglement'] = dynamics.compute_entanglement_metrics()
            metrics['bloch_vectors'] = dynamics.get_bloch_vectors()
            
            try:
                metrics['chaos'] = dynamics.compute_quantum_chaos_metrics(
                    num_perturbations=5, perturbation_strength=1e-6
                )
            except Exception as e:
                print(f"Warning: Could not compute chaos metrics: {e}")
        
        # Return combined results
        return {
            'dynamics': dynamics,
            'results': results,
            'hamiltonian': hamiltonian,
            'times': times,
            'metrics': metrics
        }
    
    @staticmethod
    def run_heisenberg_model(J_coupling=1.0, anisotropy=0.0, initial_state_type='ghz',
                           tmax=10.0, nsteps=1000, compute_metrics=True):
        """
        Run a simulation with a Heisenberg model Hamiltonian.
        
        Args:
            J_coupling (float): Strength of Heisenberg interaction
            anisotropy (float): Anisotropy parameter for the z-component
            initial_state_type (str): Type of initial state ('ghz', 'w', or 'product')
            tmax (float): Maximum simulation time
            nsteps (int): Number of time steps
            compute_metrics (bool): Whether to compute entanglement metrics
            
        Returns:
            dict: Results dictionary containing simulation results and metrics
        """
        # Create Hamiltonian
        hamiltonian_model = ThreeQubitHamiltonian()
        hamiltonian = hamiltonian_model.create_heisenberg_hamiltonian(
            J_coupling=J_coupling, anisotropy=anisotropy
        )
        
        # Set up dynamics
        dynamics = ThreeQubitDynamics(hamiltonian)
        
        # Set initial state
        if initial_state_type == 'ghz':
            dynamics.create_ghz_state()
        elif initial_state_type == 'w':
            dynamics.create_w_state()
        elif initial_state_type == 'product':
            # Create a non-symmetric product state
            plus_x = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
            minus_x = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
            plus_y = (qt.basis(2, 0) + 1j*qt.basis(2, 1)).unit()
            
            state = qt.tensor(plus_x, minus_x, plus_y)
            dynamics.set_initial_state(state)
        else:
            raise ValueError("initial_state_type must be 'ghz', 'w', or 'product'")
        
        # Run simulation
        times = np.linspace(0, tmax, nsteps)
        
        # Define expectation operators
        sx_list = [
            qt.tensor(qt.sigmax(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmax(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmax())
        ]
        
        sy_list = [
            qt.tensor(qt.sigmay(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmay(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmay())
        ]
        
        sz_list = [
            qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmaz())
        ]
        
        # Run simulation with expectation value calculations
        e_ops = sx_list + sy_list + sz_list
        results = dynamics.run_simulation(times, e_ops=e_ops)
        
        # Compute metrics if requested
        metrics = {}
        if compute_metrics:
            metrics['entanglement'] = dynamics.compute_entanglement_metrics()
            metrics['bloch_vectors'] = dynamics.get_bloch_vectors()
            
            try:
                metrics['chaos'] = dynamics.compute_quantum_chaos_metrics(
                    num_perturbations=5, perturbation_strength=1e-6
                )
            except Exception as e:
                print(f"Warning: Could not compute chaos metrics: {e}")
        
        # Return combined results
        return {
            'dynamics': dynamics,
            'results': results,
            'hamiltonian': hamiltonian,
            'times': times,
            'metrics': metrics
        }
    
    @staticmethod
    def run_gravitational_analog_model(G=1.0, decay_power=1.0, driving_strength=0.2,
                                    initial_state_type='ghz', tmax=10.0, nsteps=1000,
                                    compute_metrics=True):
        """
        Run a simulation with a gravitational analog Hamiltonian.
        
        Args:
            G (float): Strength of "gravitational" interaction
            decay_power (float): Power of decay for interaction (mimics 1/r^p)
            driving_strength (float): Strength of transverse driving field
            initial_state_type (str): Type of initial state ('ghz', 'w', or 'product')
            tmax (float): Maximum simulation time
            nsteps (int): Number of time steps
            compute_metrics (bool): Whether to compute entanglement metrics
            
        Returns:
            dict: Results dictionary containing simulation results and metrics
        """
        # Create Hamiltonian
        hamiltonian_model = ThreeQubitHamiltonian()
        hamiltonian = hamiltonian_model.create_gravitational_analog_hamiltonian(
            G=G, decay_power=decay_power, driving_strength=driving_strength
        )
        
        # Set up dynamics
        dynamics = ThreeQubitDynamics(hamiltonian)
        
        # Set initial state
        if initial_state_type == 'ghz':
            dynamics.create_ghz_state()
        elif initial_state_type == 'w':
            dynamics.create_w_state()
        elif initial_state_type == 'product':
            # Create a product state with unequal "masses"
            # Weighted superpositions can simulate different "masses"
            alpha = 0.8  # Larger "mass" for qubit 1
            beta = 0.6   # Medium "mass" for qubit 2
            gamma = 0.4  # Smaller "mass" for qubit 3
            
            state1 = (alpha * qt.basis(2, 0) + np.sqrt(1-alpha**2) * qt.basis(2, 1)).unit()
            state2 = (beta * qt.basis(2, 0) + np.sqrt(1-beta**2) * qt.basis(2, 1)).unit()
            state3 = (gamma * qt.basis(2, 0) + np.sqrt(1-gamma**2) * qt.basis(2, 1)).unit()
            
            state = qt.tensor(state1, state2, state3)
            dynamics.set_initial_state(state)
        else:
            raise ValueError("initial_state_type must be 'ghz', 'w', or 'product'")
        
        # Run simulation
        times = np.linspace(0, tmax, nsteps)
        
        # Define expectation operators
        sx_list = [
            qt.tensor(qt.sigmax(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmax(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmax())
        ]
        
        sy_list = [
            qt.tensor(qt.sigmay(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmay(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmay())
        ]
        
        sz_list = [
            qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmaz())
        ]
        
        # Run simulation with expectation value calculations
        e_ops = sx_list + sy_list + sz_list
        results = dynamics.run_simulation(times, e_ops=e_ops)
        
        # Compute metrics if requested
        metrics = {}
        if compute_metrics:
            metrics['entanglement'] = dynamics.compute_entanglement_metrics()
            metrics['bloch_vectors'] = dynamics.get_bloch_vectors()
            
            try:
                metrics['chaos'] = dynamics.compute_quantum_chaos_metrics(
                    num_perturbations=5, perturbation_strength=1e-6
                )
            except Exception as e:
                print(f"Warning: Could not compute chaos metrics: {e}")
        
        # Return combined results
        return {
            'dynamics': dynamics,
            'results': results,
            'hamiltonian': hamiltonian,
            'times': times,
            'metrics': metrics
        }
    
    @staticmethod
    def run_time_dependent_model(base_hamiltonian_type='ising', modulation_type='sinusoidal',
                               amplitude=0.5, frequency=1.0, phase=0.0,
                               initial_state_type='ghz', tmax=10.0, nsteps=1000,
                               compute_metrics=True, **hamiltonian_params):
        """
        Run a simulation with a time-dependent Hamiltonian.
        
        Args:
            base_hamiltonian_type (str): Type of base Hamiltonian ('ising', 'heisenberg', 'gravitational')
            modulation_type (str): Type of time modulation ('sinusoidal', 'square', 'sawtooth')
            amplitude (float): Amplitude of the modulation
            frequency (float): Frequency of the modulation
            phase (float): Phase offset of the modulation
            initial_state_type (str): Type of initial state ('ghz', 'w', or 'product')
            tmax (float): Maximum simulation time
            nsteps (int): Number of time steps
            compute_metrics (bool): Whether to compute entanglement metrics
            **hamiltonian_params: Additional parameters for the base Hamiltonian
            
        Returns:
            dict: Results dictionary containing simulation results and metrics
        """
        # Create base Hamiltonian
        hamiltonian_model = ThreeQubitHamiltonian()
        
        if base_hamiltonian_type == 'ising':
            J_coupling = hamiltonian_params.get('J_coupling', 1.0)
            h_field = hamiltonian_params.get('h_field', 0.5)
            base_hamiltonian = hamiltonian_model.create_ising_hamiltonian(
                J_coupling=J_coupling, h_field=h_field
            )
        elif base_hamiltonian_type == 'heisenberg':
            J_coupling = hamiltonian_params.get('J_coupling', 1.0)
            anisotropy = hamiltonian_params.get('anisotropy', 0.0)
            base_hamiltonian = hamiltonian_model.create_heisenberg_hamiltonian(
                J_coupling=J_coupling, anisotropy=anisotropy
            )
        elif base_hamiltonian_type == 'gravitational':
            G = hamiltonian_params.get('G', 1.0)
            decay_power = hamiltonian_params.get('decay_power', 1.0)
            driving_strength = hamiltonian_params.get('driving_strength', 0.2)
            base_hamiltonian = hamiltonian_model.create_gravitational_analog_hamiltonian(
                G=G, decay_power=decay_power, driving_strength=driving_strength
            )
        else:
            raise ValueError("base_hamiltonian_type must be 'ising', 'heisenberg', or 'gravitational'")
        
        # Select modulation function
        if modulation_type == 'sinusoidal':
            from .hamiltonian import sinusoidal_modulation
            modulation_func = sinusoidal_modulation
        else:
            # For custom modulation types, define corresponding functions in hamiltonian.py
            from .hamiltonian import sinusoidal_modulation
            modulation_func = sinusoidal_modulation
            print(f"Warning: Unknown modulation type '{modulation_type}'. Using sinusoidal modulation.")
        
        # Create time-dependent Hamiltonian
        args = {'amplitude': amplitude, 'frequency': frequency, 'phase': phase}
        hamiltonian = hamiltonian_model.create_time_dependent_hamiltonian(
            base_hamiltonian, modulation_func, args
        )
        
        # Set up dynamics
        dynamics = ThreeQubitDynamics(hamiltonian)
        
        # Set initial state
        if initial_state_type == 'ghz':
            dynamics.create_ghz_state()
        elif initial_state_type == 'w':
            dynamics.create_w_state()
        elif initial_state_type == 'product':
            dynamics.set_initial_state()
        else:
            raise ValueError("initial_state_type must be 'ghz', 'w', or 'product'")
        
        # Run simulation
        times = np.linspace(0, tmax, nsteps)
        
        # Define expectation operators
        sx_list = [
            qt.tensor(qt.sigmax(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmax(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmax())
        ]
        
        sy_list = [
            qt.tensor(qt.sigmay(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmay(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmay())
        ]
        
        sz_list = [
            qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2)),
            qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmaz())
        ]
        
        # Run simulation with expectation value calculations
        e_ops = sx_list + sy_list + sz_list
        results = dynamics.run_simulation(times, e_ops=e_ops)
        
        # Compute metrics if requested
        metrics = {}
        if compute_metrics:
            metrics['entanglement'] = dynamics.compute_entanglement_metrics()
            metrics['bloch_vectors'] = dynamics.get_bloch_vectors()
            
            # Note: Chaos metrics may not be applicable for time-dependent systems
            # as implemented, but we can still try to compute them
            try:
                metrics['chaos'] = dynamics.compute_quantum_chaos_metrics(
                    num_perturbations=5, perturbation_strength=1e-6
                )
            except Exception as e:
                print(f"Warning: Could not compute chaos metrics: {e}")
        
        # Return combined results
        return {
            'dynamics': dynamics,
            'results': results,
            'hamiltonian': base_hamiltonian,  # Return the base Hamiltonian for analysis
            'times': times,
            'metrics': metrics,
            'modulation_args': args
        }
