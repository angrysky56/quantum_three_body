"""
Quantum Hamiltonian module for the three-qubit system.

This module provides classes and functions to create and manipulate 
Hamiltonians that model three interacting qubits, designed to mimic 
aspects of the classical three-body problem.
"""

import numpy as np
import qutip as qt


class ThreeQubitHamiltonian:
    """
    A class representing a Hamiltonian for three interacting qubits.
    
    This class creates Hamiltonians that can model various types of 
    interactions between three qubits, potentially mimicking aspects 
    of the classical three-body gravitational problem.
    
    Attributes:
        num_qubits (int): Number of qubits in the system (fixed at 3)
        dims (list): Dimensions of each subsystem [2, 2, 2]
        hamiltonian (Qobj): The QuTiP quantum object representing the Hamiltonian
    """
    
    def __init__(self):
        """Initialize with a default 3-qubit system."""
        self.num_qubits = 3
        self.dims = [2, 2, 2]
        self.hamiltonian = None
    
    def create_ising_hamiltonian(self, J_coupling=1.0, h_field=0.5):
        """
        Create an Ising model Hamiltonian with transverse field.
        
        H = J * (σ_z^1 ⊗ σ_z^2 + σ_z^2 ⊗ σ_z^3 + σ_z^3 ⊗ σ_z^1) + h * (σ_x^1 + σ_x^2 + σ_x^3)
        
        Args:
            J_coupling (float): Strength of Ising interaction
            h_field (float): Strength of transverse field
            
        Returns:
            Qobj: The Ising model Hamiltonian
        """
        # Pauli matrices
        sx = qt.sigmax()
        sz = qt.sigmaz()
        
        # Identity operator
        id_op = qt.qeye(2)
        
        # Interaction terms
        interaction_12 = J_coupling * qt.tensor(sz, sz, id_op)
        interaction_23 = J_coupling * qt.tensor(id_op, sz, sz)
        interaction_31 = J_coupling * qt.tensor(sz, id_op, sz)
        
        # Transverse field terms
        field_1 = h_field * qt.tensor(sx, id_op, id_op)
        field_2 = h_field * qt.tensor(id_op, sx, id_op)
        field_3 = h_field * qt.tensor(id_op, id_op, sx)
        
        # Total Hamiltonian
        self.hamiltonian = interaction_12 + interaction_23 + interaction_31 + field_1 + field_2 + field_3
        
        return self.hamiltonian
    
    def create_heisenberg_hamiltonian(self, J_coupling=1.0, anisotropy=0.0):
        """
        Create a Heisenberg model Hamiltonian with optional anisotropy.
        
        H = J * (σ_x^1 ⊗ σ_x^2 + σ_y^1 ⊗ σ_y^2 + σ_z^1 ⊗ σ_z^2 + ...) + Δ * (σ_z^1 ⊗ σ_z^2 + ...)
        
        Args:
            J_coupling (float): Strength of Heisenberg interaction
            anisotropy (float): Anisotropy parameter (Δ) for the z-component
            
        Returns:
            Qobj: The Heisenberg model Hamiltonian
        """
        # Pauli matrices
        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()
        
        # Identity operator
        id_op = qt.qeye(2)
        
        # XY-plane interactions
        xy_interaction_12 = J_coupling * (qt.tensor(sx, sx, id_op) + qt.tensor(sy, sy, id_op))
        xy_interaction_23 = J_coupling * (qt.tensor(id_op, sx, sx) + qt.tensor(id_op, sy, sy))
        xy_interaction_31 = J_coupling * (qt.tensor(sx, id_op, sx) + qt.tensor(sy, id_op, sy))
        
        # Z-axis interactions (with anisotropy)
        z_coupling = J_coupling * (1 + anisotropy)
        z_interaction_12 = z_coupling * qt.tensor(sz, sz, id_op)
        z_interaction_23 = z_coupling * qt.tensor(id_op, sz, sz)
        z_interaction_31 = z_coupling * qt.tensor(sz, id_op, sz)
        
        # Total Hamiltonian
        self.hamiltonian = (xy_interaction_12 + xy_interaction_23 + xy_interaction_31 +
                           z_interaction_12 + z_interaction_23 + z_interaction_31)
        
        return self.hamiltonian
    
    def create_gravitational_analog_hamiltonian(self, G=1.0, decay_power=1.0, driving_strength=0.2):
        """
        Create a Hamiltonian that attempts to mimic gravitational-like interactions.
        
        H = -G * (σ_z^1 ⊗ σ_z^2 / r^p + σ_z^2 ⊗ σ_z^3 / r^p + σ_z^3 ⊗ σ_z^1 / r^p) + 
            driving_strength * (σ_x^1 + σ_x^2 + σ_x^3)
            
        Where r^p is simulated by the decay_power parameter.
        
        Args:
            G (float): Strength of "gravitational" interaction
            decay_power (float): Power of decay for interaction (mimics 1/r^p)
            driving_strength (float): Strength of transverse driving field
            
        Returns:
            Qobj: The gravitational analog Hamiltonian
        """
        # Pauli matrices
        sx = qt.sigmax()
        sz = qt.sigmaz()
        
        # Identity operator
        id_op = qt.qeye(2)
        
        # We'll use a simplified model where the "gravitational" coupling is tuned 
        # by the decay_power parameter instead of a true spatial model
        
        # Attraction terms (negative sign for attraction)
        attraction_12 = -G * qt.tensor(sz, sz, id_op)
        attraction_23 = -G * qt.tensor(id_op, sz, sz)
        attraction_31 = -G * qt.tensor(sz, id_op, sz)
        
        # Driving field terms to provide energy/momentum
        drive_1 = driving_strength * qt.tensor(sx, id_op, id_op)
        drive_2 = driving_strength * qt.tensor(id_op, sx, id_op)
        drive_3 = driving_strength * qt.tensor(id_op, id_op, sx)
        
        # Total Hamiltonian
        self.hamiltonian = (attraction_12 + attraction_23 + attraction_31 +
                           drive_1 + drive_2 + drive_3)
        
        return self.hamiltonian
    
    def create_time_dependent_hamiltonian(self, base_hamiltonian, modulation_func, args=None):
        """
        Create a time-dependent Hamiltonian based on a modulation function.
        
        Args:
            base_hamiltonian (Qobj): Base Hamiltonian to modulate
            modulation_func (callable): Function that takes time t and returns a coefficient
            args (dict): Additional arguments for the modulation function
            
        Returns:
            list: QuTiP format time-dependent Hamiltonian [H0, [H1, func1], [H2, func2], ...]
        """
        if args is None:
            args = {}
            
        # Time-dependent Hamiltonian in QuTiP format
        self.hamiltonian = [base_hamiltonian, [base_hamiltonian, modulation_func]]
        
        return self.hamiltonian


def sinusoidal_modulation(t, args):
    """
    A sinusoidal modulation function for time-dependent Hamiltonians.
    
    Args:
        t (float): Time point
        args (dict): Arguments including 'amplitude' and 'frequency'
        
    Returns:
        float: Modulation coefficient at time t
    """
    amplitude = args.get('amplitude', 0.5)
    frequency = args.get('frequency', 1.0)
    phase = args.get('phase', 0.0)
    
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)
