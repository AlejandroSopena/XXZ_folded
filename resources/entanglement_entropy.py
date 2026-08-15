# Python 3.10.18
from jax import config as jax_config
jax_config.update("jax_platforms", "cpu")  

import jax
jax.config.update("jax_enable_x64", True)
print(jax.default_backend(), jax.devices())

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

from pathlib import Path

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titlesize": 13
})

# Base directory and data directory
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
data_dir = BASE_DIR / "data"

# Load the data
with open(data_dir / "states.pkl", "rb") as f:
    states = pickle.load(f)

def entropy(states):
    """
    Compute the von Neumann entanglement entropy of a set of D quantum states |Ψ_a⟩ over N qubits across all connected bipartitions.
    The entanglement entropy of |Ψ_a⟩ across the bipartition of 

        S_k = - ∑_i λ_i**2 log_2 λ_i**2

    where λ_i denote the singular values of |Ψ_a⟩ along the bipartition. 
    To compute λ_i for each bipartition of N qubits into a first block of k adjacent qubits and a second of N − k, 
    |Ψ_a⟩ is rephrased as a matrix of shape (2**k, 2**(N − k)).


    Parameters
    ----------
    states : jnp.ndarray
        Array of shape (2**N, D), where each column corresponds to a distinct state vector |Ψ_a⟩.

    Returns
    -------
    results : dict
        Dictionary with keys:
            - "N": number of qubits.
            - "entropy_profiles": list of D elements, where each element is [S_1, S_2, ..., S_{N - 1}] for the corresponding state |Ψ_a⟩.
    """

    # Test if the input represents states over N qubits
    total_aim, D = states.shape
    N = int(jnp.round(jnp.log2(total_aim)))
    if states.shape[0] != 2**N:
        raise ValueError(
            f"The first dimension must be 2**N, but received {states.shape[0]}."
        )

    # Initialise list to store the entropy profile for each state
    entropy_profiles = []

    # Loop over all D input state
    for d in range(D):

        # Extract the d-th state
        state = states[:, d]

        # Initialise list to store the entropy value for each bipartition
        entropy_profile = []

        # Loop over all connected bipartitions of N qubits
        for k in range(1, N):

            # Reshape the input into a state of shape (2**k, 2**(N−k))
            bipartite_matrix = jnp.reshape(state, (2**k, -1))

            # Compute the singular values along the bipartition λ_i
            singular_values = jnp.linalg.svd(bipartite_matrix, compute_uv=False)

            # Square the singular values λ_i**2 
            squared_values = singular_values ** 2

            # Compute the entropy
            entropy_value = -jnp.sum(jnp.where(squared_values > 0, squared_values * jnp.log2(squared_values), 0.0)).real

            # Append the outcome to the list
            entropy_profile.append(entropy_value)

        # Append the profile for this state
        entropy_profiles.append(entropy_profile)

    # Return the number of qubits and a list of collections of profiles
    return {"N": N, "entropy_profiles": entropy_profiles}

def plot_entropy(N, entropy_profiles):
    """
    Plot the von Neumann entanglement entropy of a set of D quantum states |Ψ_a⟩ over N qubits across all connected bipartitions.

    Parameters
    ----------
    N : int
        Number of qubits.

    entropy_profiles : list
        List of D entropy-profile lists [S_1, S_2, ..., S_{N - 1}] across all connected bipartitions,
        as returned by the function `entropy`.

    Returns
    -------
    None

    """
    # Begin a new figure for this value of N
    plt.figure(figsize=(6, 4))

    # Define the independent variable: the number of qubits in the left block
    partition_sizes = list(range(1, N))

    # Plot one or several entropy profiles in a single command
    plt.plot(partition_sizes, jnp.atleast_2d(entropy_profiles).T)

    # Tick
    if N % 2 == 0:   
        plt.xticks(range(1, N, 2))
    else:
        plt.xticks(range(1, N))

    # Set title
    plt.title(rf"$N = {N}$ qubits")

    # Label the axes
    plt.xlabel("Qubits of the partition")
    plt.ylabel("Entropy ${S}$")

    # Add grid
    plt.grid(True)

    # Display the figure
    plt.show()

def extremal_entropy(dictionary_entropies, mode_numbers, percentage=0.05):
    """
    Identify, 
    from a collection of profiles of the von Neumann entanglement entropy across all connected bipartitions, 
    the fraction of Bethe states exhibiting extremal mean entropy.

    The function now receives the set of quantum–number labels directly as an array `mode_numbers` of shape (M, D),
    where each column lists the mode numbers labelling a Bethe state. 
    Column indices are assumed to match the ordering of entropy profiles in `dictionary_entropies`.

    Parameters
    ----------
    dictionary_entropies : dict
        Dictionary returned by `entropy`, with keys:
            - "N": number of qubits;
            - "entropy_profiles": list of D lists storing [S₁, …, S_{N−1}] for each state.

    mode_numbers : array-like
        A jax.numpy or numpy array of shape (M, D), where each column lists 
        the mode numbers indexing the corresponding Bethe state.

    percentage : float, optional
        Fraction of states to extract at the top and bottom of the entropy distribution.
        Defaults to 0.05.

    Returns
    -------
    None
    """
    # Number of qubits
    N = dictionary_entropies["N"]

    # Convert labels to JAX array if needed
    mode_numbers = jnp.array(mode_numbers, dtype=jnp.int32)

    # Extract entropy profiles and number of states
    entropy_profiles = dictionary_entropies["entropy_profiles"]
    D = len(entropy_profiles)

    # Verify compatibility
    M, D_check = mode_numbers.shape
    if int(D_check) != int(D):
        raise ValueError(
            f"Inconsistent data: entropy dictionary contains D = {D} states, "
            f"whereas the array of mode numbers contains D = {D_check} states."
        )

    # Compute mean entropies using pure jnp
    mean_entropies = jnp.array(
        [jnp.mean(jnp.array(profile, dtype=jnp.float64))
         for profile in entropy_profiles],
        dtype=jnp.float64
    )

    # Number of states to extract
    count = max(1, int(percentage * D))

    # Host ordering
    mean_entropies_host = np.array(mean_entropies, dtype=float)

    # Sorting of indices
    sorted_indices = np.argsort(mean_entropies_host)

    # Extremal sectors
    lowest_indices  = sorted_indices[:count]
    highest_indices = sorted_indices[-count:]

    # Determine header width according to the number of digits in N−2 and M
    digits_N = len(str(N - 2))
    digits_M = len(str(M))

    # Select width
    if digits_N == 1 and digits_M == 1:
        width = 68
    elif digits_N == 2 and digits_M == 2:
        width = 70
    else:
        width = 69

    # Define top and bottom headers
    header_top    = "\n" + "x" * width
    header_bottom = "x"  * width + "\n"

    # Header
    print(header_top)
    print(f"N = {N - 2} qubits in the bulk, Hamming weight M = {M}, and no domain walls.")
    print(header_bottom)

    if M == 1:
        print(f"The admissible mode numbers are 1 ≤ n ≤ {N - M - 1}" + "\n")

    elif M == 2:
        print(f"The admissible mode numbers are 1 ≤ n_1 < n_2 ≤ {N - M - 1}" + "\n")

    else:
        print(f"The admissible mode numbers are 1 ≤ n_1 <...< n_{M} ≤ {N - M - 1}" + "\n")

    # Header: highest-entropy states
    print("x"*95)
    print("Bethe states of highest average von Neumann entanglement entropy across connected bipartitions.")
    print("x"*95 + "\n")

    for idx in highest_indices:

        # Mode numbers
        mode_nums = tuple(int(x) for x in mode_numbers[:, idx])

        # Mean entropy
        S_avg = float(mean_entropies_host[idx])

        # Pretty-print
        mode_nums_str = ", ".join(str(q) for q in mode_nums)

        print(
            f"The Bethe state labelled by mode numbers {mode_nums_str} "
            f"has an average entanglement entropy "
            f"S_average = {S_avg:.6f}"
        )

        print()

    # Header: lowest-entropy states
    print("x"*94)
    print("Bethe states of lowest average von Neumann entanglement entropy across connected bipartitions.")
    print("x"*94 + "\n")

    for idx in lowest_indices:

        mode_nums = tuple(int(x) for x in mode_numbers[:, idx])
        S_avg = float(mean_entropies_host[idx])

        mode_nums_str = ", ".join(str(q) for q in mode_nums)

        print(
            f"The Bethe state labelled by mode numbers {mode_nums_str} "
            f"has an average entanglement entropy "
            f"S_average = {S_avg:.6f}"
        )

        print()

    return
    
# Compute the von Neumann entanglement entropy profiles
dictionary_entropies = entropy(states)

# Plot the von Neumann entanglement entropy profiles
plot_entropy(dictionary_entropies["N"], dictionary_entropies["entropy_profiles"])

# Load the data
with open(data_dir / "mode_numbers.pkl", "rb") as f:
    mode_numbers = pickle.load(f)

# Compute the fraction of Bethe states inside the eigenspace with extremal mean Von Neumann entanglement entropy across bipartitions, 
# if present
extremal_entropy(dictionary_entropies, mode_numbers)

