"""
Visualization module for the three-qubit system.

This module provides functions to visualize the dynamics, entanglement,
and other properties of the three-qubit system.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union

# Set seaborn style for better visuals
sns.set(style="whitegrid")


def plot_bloch_sphere(ax, vectors, qubit_idx, color='b', alpha=0.7, tail_fraction=0.0):
    """
    Plot Bloch vectors on a 3D Bloch sphere.
    
    Args:
        ax (Axes3D): Matplotlib 3D axis
        vectors (dict): Bloch vectors from dynamics.get_bloch_vectors()
        qubit_idx (int): Index of the qubit (1, 2, or 3)
        color (str): Color of the trajectory
        alpha (float): Alpha transparency
        tail_fraction (float): Fraction of trajectory to plot (0.0 plots all)
    """
    qubit_key = f'qubit{qubit_idx}'
    
    # Extract vectors
    x = vectors[qubit_key]['x']
    y = vectors[qubit_key]['y']
    z = vectors[qubit_key]['z']
    
    # Determine start index based on tail_fraction
    if tail_fraction > 0:
        start_idx = max(0, int((1 - tail_fraction) * len(x)))
        x = x[start_idx:]
        y = y[start_idx:]
        z = z[start_idx:]
    
    # Plot the trajectory
    ax.plot(x, y, z, color=color, alpha=alpha, lw=2)
    
    # Plot the final point
    ax.scatter(x[-1], y[-1], z[-1], color=color, s=50)


def create_bloch_sphere(ax):
    """
    Create a Bloch sphere on the given 3D axis.
    
    Args:
        ax (Axes3D): Matplotlib 3D axis
    """
    # Draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)
    
    # Draw axes
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='g', arrow_length_ratio=0.1, label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='b', arrow_length_ratio=0.1, label='Z')
    
    # Add basis state indicators
    ax.text(1.1, 0, 0, "|+⟩", color='r')
    ax.text(-1.1, 0, 0, "|−⟩", color='r')
    ax.text(0, 1.1, 0, "|+i⟩", color='g')
    ax.text(0, -1.1, 0, "|−i⟩", color='g')
    ax.text(0, 0, 1.1, "|0⟩", color='b')
    ax.text(0, 0, -1.1, "|1⟩", color='b')
    
    # Set limits and labels
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_three_qubit_bloch_spheres(bloch_vectors, title="Three-Qubit Bloch Vectors"):
    """
    Create a figure with three Bloch spheres, one for each qubit.
    
    Args:
        bloch_vectors (dict): Bloch vectors from dynamics.get_bloch_vectors()
        title (str): Title for the figure
        
    Returns:
        fig (Figure): Matplotlib figure object
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Create three Bloch spheres
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Add Bloch sphere wireframes
    create_bloch_sphere(ax1)
    create_bloch_sphere(ax2)
    create_bloch_sphere(ax3)
    
    # Plot trajectories for each qubit
    plot_bloch_sphere(ax1, bloch_vectors, 1, color='r')
    plot_bloch_sphere(ax2, bloch_vectors, 2, color='g')
    plot_bloch_sphere(ax3, bloch_vectors, 3, color='b')
    
    # Set titles
    ax1.set_title("Qubit 1", fontsize=14)
    ax2.set_title("Qubit 2", fontsize=14)
    ax3.set_title("Qubit 3", fontsize=14)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def animate_bloch_spheres(bloch_vectors, interval=50, tail_fraction=0.2):
    """
    Create an animation of the Bloch vectors over time.
    
    Args:
        bloch_vectors (dict): Bloch vectors from dynamics.get_bloch_vectors()
        interval (int): Time between frames in milliseconds
        tail_fraction (float): Fraction of the trajectory to show as "tail"
        
    Returns:
        anim (FuncAnimation): Matplotlib animation object
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Create three Bloch spheres
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Setup initial state
    create_bloch_sphere(ax1)
    create_bloch_sphere(ax2)
    create_bloch_sphere(ax3)
    
    # Extract data
    times = bloch_vectors['times']
    qubit1_x = bloch_vectors['qubit1']['x']
    qubit1_y = bloch_vectors['qubit1']['y']
    qubit1_z = bloch_vectors['qubit1']['z']
    
    qubit2_x = bloch_vectors['qubit2']['x']
    qubit2_y = bloch_vectors['qubit2']['y']
    qubit2_z = bloch_vectors['qubit2']['z']
    
    qubit3_x = bloch_vectors['qubit3']['x']
    qubit3_y = bloch_vectors['qubit3']['y']
    qubit3_z = bloch_vectors['qubit3']['z']
    
    # Set titles
    ax1.set_title("Qubit 1", fontsize=14)
    ax2.set_title("Qubit 2", fontsize=14)
    ax3.set_title("Qubit 3", fontsize=14)
    fig.suptitle("Three-Qubit Dynamics", fontsize=16)
    
    # Get line objects
    line1, = ax1.plot([], [], [], 'r-', lw=2)
    point1 = ax1.scatter([], [], [], 'ro', s=50)
    
    line2, = ax2.plot([], [], [], 'g-', lw=2)
    point2 = ax2.scatter([], [], [], 'go', s=50)
    
    line3, = ax3.plot([], [], [], 'b-', lw=2)
    point3 = ax3.scatter([], [], [], 'bo', s=50)
    
    # Time indicator text
    time_text = ax1.text2D(0.05, 0.95, '', transform=ax1.transAxes)
    
    def init():
        line1.set_data([], [])
        line1.set_3d_properties([])
        point1._offsets3d = ([], [], [])
        
        line2.set_data([], [])
        line2.set_3d_properties([])
        point2._offsets3d = ([], [], [])
        
        line3.set_data([], [])
        line3.set_3d_properties([])
        point3._offsets3d = ([], [], [])
        
        time_text.set_text('')
        
        return line1, point1, line2, point2, line3, point3, time_text
    
    def animate(i):
        # Calculate how many points to show in the tail
        tail_len = int(tail_fraction * i) if i > 0 else 1
        start_idx = max(0, i - tail_len)
        
        # Update qubit 1
        line1.set_data(qubit1_x[start_idx:i+1], qubit1_y[start_idx:i+1])
        line1.set_3d_properties(qubit1_z[start_idx:i+1])
        point1._offsets3d = ([qubit1_x[i]], [qubit1_y[i]], [qubit1_z[i]])
        
        # Update qubit 2
        line2.set_data(qubit2_x[start_idx:i+1], qubit2_y[start_idx:i+1])
        line2.set_3d_properties(qubit2_z[start_idx:i+1])
        point2._offsets3d = ([qubit2_x[i]], [qubit2_y[i]], [qubit2_z[i]])
        
        # Update qubit 3
        line3.set_data(qubit3_x[start_idx:i+1], qubit3_y[start_idx:i+1])
        line3.set_3d_properties(qubit3_z[start_idx:i+1])
        point3._offsets3d = ([qubit3_x[i]], [qubit3_y[i]], [qubit3_z[i]])
        
        # Update time text
        time_text.set_text(f'Time: {times[i]:.2f}')
        
        return line1, point1, line2, point2, line3, point3, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(times), interval=interval,
                                   blit=True)
    
    plt.tight_layout()
    return anim


def plot_entanglement_metrics(metrics, title="Entanglement Metrics"):
    """
    Plot entanglement metrics over time.
    
    Args:
        metrics (dict): Entanglement metrics from dynamics.compute_entanglement_metrics()
        title (str): Title for the figure
        
    Returns:
        fig (Figure): Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot pairwise concurrence
    axes[0, 0].plot(metrics['times'], metrics['concurrence_12'], 'r-', label='Qubits 1-2')
    axes[0, 0].plot(metrics['times'], metrics['concurrence_23'], 'g-', label='Qubits 2-3')
    axes[0, 0].plot(metrics['times'], metrics['concurrence_13'], 'b-', label='Qubits 1-3')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Concurrence')
    axes[0, 0].set_title('Pairwise Entanglement (Concurrence)')
    axes[0, 0].legend()
    
    # Plot von Neumann entropy for each qubit
    axes[0, 1].plot(metrics['times'], metrics['entropy_1'], 'r-', label='Qubit 1')
    axes[0, 1].plot(metrics['times'], metrics['entropy_2'], 'g-', label='Qubit 2')
    axes[0, 1].plot(metrics['times'], metrics['entropy_3'], 'b-', label='Qubit 3')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('von Neumann Entropy')
    axes[0, 1].set_title('Single-Qubit Entropy')
    axes[0, 1].legend()
    
    # Plot tripartite entanglement
    axes[1, 0].plot(metrics['times'], metrics['tripartite_entanglement'], 'k-')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Average Entropy')
    axes[1, 0].set_title('Tripartite Entanglement Measure')
    
    # Plot total entanglement (sum of pairwise concurrences)
    total_concurrence = (metrics['concurrence_12'] + 
                         metrics['concurrence_23'] + 
                         metrics['concurrence_13'])
    axes[1, 1].plot(metrics['times'], total_concurrence, 'purple')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Sum of Concurrences')
    axes[1, 1].set_title('Total Pairwise Entanglement')
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_quantum_chaos_metrics(metrics, title="Quantum Chaos Metrics"):
    """
    Plot quantum chaos metrics over time.
    
    Args:
        metrics (dict): Quantum chaos metrics from dynamics.compute_quantum_chaos_metrics()
        title (str): Title for the figure
        
    Returns:
        fig (Figure): Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot average fidelity decay
    axes[0].plot(metrics['times'], metrics['avg_fidelity_decay'], 'b-')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Average Fidelity')
    axes[0].set_title('Quantum State Fidelity Decay')
    
    # Plot estimated Lyapunov exponent (skip the first point which may be undefined)
    valid_indices = ~np.isnan(metrics['lyapunov_estimate'][1:]) & ~np.isinf(metrics['lyapunov_estimate'][1:])
    if np.any(valid_indices):
        axes[1].plot(metrics['times'][1:][valid_indices], 
                    metrics['lyapunov_estimate'][1:][valid_indices], 'r-')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Estimated Lyapunov Exponent')
        axes[1].set_title('Quantum Lyapunov Exponent Estimate')
    else:
        axes[1].text(0.5, 0.5, 'No valid Lyapunov exponent estimates', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[1].transAxes)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_state_populations(results, title="Quantum State Populations"):
    """
    Plot the populations of different basis states over time.
    
    Args:
        results (Result): Results from dynamics.run_simulation()
        title (str): Title for the figure
        
    Returns:
        fig (Figure): Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    times = results.times
    states = results.states
    
    # Basis states for 3 qubits
    basis_states = [
        '|000⟩', '|001⟩', '|010⟩', '|011⟩', 
        '|100⟩', '|101⟩', '|110⟩', '|111⟩'
    ]
    
    # Calculate probabilities for each basis state
    probabilities = np.zeros((len(times), 8))
    
    for i, state in enumerate(states):
        for j in range(8):
            # Create basis state in the computational basis
            basis = np.zeros(8)
            basis[j] = 1
            basis_ket = qt.Qobj(basis).reshape((-1, 1))
            
            # Calculate probability amplitude
            amplitude = state.overlap(basis_ket)
            probabilities[i, j] = np.abs(amplitude)**2
    
    # Plot probabilities
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for j in range(8):
        ax.plot(times, probabilities[:, j], label=basis_states[j], color=colors[j])
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_energy_evolution(results, hamiltonian, title="System Energy Evolution"):
    """
    Plot the energy of the system over time.
    
    Args:
        results (Result): Results from dynamics.run_simulation()
        hamiltonian (Qobj): Time-independent Hamiltonian of the system
        title (str): Title for the figure
        
    Returns:
        fig (Figure): Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    times = results.times
    states = results.states
    
    # Calculate energy at each time
    energy = np.zeros(len(times))
    for i, state in enumerate(states):
        energy[i] = qt.expect(hamiltonian, state)
    
    # Plot energy
    ax.plot(times, energy, 'k-')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_poincare_section(bloch_vectors, qubit_idx=1, plane='xy', 
                          title="Poincaré Section"):
    """
    Plot a Poincaré section for a qubit's Bloch vector.
    
    This function creates a pseudo-Poincaré section by plotting one component
    of the Bloch vector against another when a third component crosses zero.
    
    Args:
        bloch_vectors (dict): Bloch vectors from dynamics.get_bloch_vectors()
        qubit_idx (int): Index of the qubit (1, 2, or 3)
        plane (str): Plane to plot ('xy', 'yz', or 'xz')
        title (str): Title for the figure
        
    Returns:
        fig (Figure): Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    qubit_key = f'qubit{qubit_idx}'
    
    # Extract vectors
    x = bloch_vectors[qubit_key]['x']
    y = bloch_vectors[qubit_key]['y']
    z = bloch_vectors[qubit_key]['z']
    
    # Determine components based on plane
    if plane == 'xy':
        comp1, comp2, crossing = x, y, z
        comp1_label, comp2_label = 'x', 'y'
    elif plane == 'yz':
        comp1, comp2, crossing = y, z, x
        comp1_label, comp2_label = 'y', 'z'
    elif plane == 'xz':
        comp1, comp2, crossing = x, z, y
        comp1_label, comp2_label = 'x', 'z'
    else:
        raise ValueError("plane must be 'xy', 'yz', or 'xz'")
    
    # Find crossing points (where the third component changes sign)
    crossing_points = []
    for i in range(1, len(crossing)):
        if crossing[i-1] * crossing[i] <= 0:  # Sign change or zero crossing
            # Linear interpolation to find precise crossing
            t = crossing[i-1] / (crossing[i-1] - crossing[i])
            cross_comp1 = comp1[i-1] + t * (comp1[i] - comp1[i-1])
            cross_comp2 = comp2[i-1] + t * (comp2[i] - comp2[i-1])
            crossing_points.append((cross_comp1, cross_comp2))
    
    # Extract points
    if crossing_points:
        points = np.array(crossing_points)
        ax.scatter(points[:, 0], points[:, 1], c='b', s=30, alpha=0.7)
    
    # Plot circle representing the Bloch sphere boundary in this plane
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    ax.set_xlabel(f'{comp1_label} Component')
    ax.set_ylabel(f'{comp2_label} Component')
    ax.set_title(f"{title} ({comp1_label}{comp2_label}-plane, Qubit {qubit_idx})")
    
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def create_phase_space_3d(bloch_vectors, title="Three-Qubit Phase Space"):
    """
    Create a 3D visualization of the combined phase space of three qubits.
    
    This function creates a 3D plot where each axis represents one qubit's
    z-component, providing a view of the combined "phase space" of the system.
    
    Args:
        bloch_vectors (dict): Bloch vectors from dynamics.get_bloch_vectors()
        title (str): Title for the figure
        
    Returns:
        fig (Figure): Matplotlib figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract z-components
    z1 = bloch_vectors['qubit1']['z']
    z2 = bloch_vectors['qubit2']['z']
    z3 = bloch_vectors['qubit3']['z']
    
    # Plot trajectory
    ax.plot(z1, z2, z3, 'b-', lw=2, alpha=0.7)
    
    # Plot initial and final points
    ax.scatter(z1[0], z2[0], z3[0], c='g', s=100, label='Initial')
    ax.scatter(z1[-1], z2[-1], z3[-1], c='r', s=100, label='Final')
    
    # Set labels and limits
    ax.set_xlabel('Qubit 1 (z)')
    ax.set_ylabel('Qubit 2 (z)')
    ax.set_zlabel('Qubit 3 (z)')
    
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    
    ax.set_title(title)
    ax.legend()
    
    return fig


# Import this at the end to avoid circular imports
try:
    import qutip as qt
except ImportError:
    print("QuTiP not found. Some plotting functions may not work.")
