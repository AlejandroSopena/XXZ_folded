# Python 3.10.18
from jax import config as jax_config
jax_config.update("jax_platforms", "cpu")  # set before importing jax

import jax
jax.config.update("jax_enable_x64", True)
print(jax.default_backend(), jax.devices())

import jax.numpy as jnp
import itertools as it
import numpy as np
import pickle
import string

from pathlib import Path

# Base directory and data directory
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
data_dir = BASE_DIR / "data"

# Load the NumPy data
with open(data_dir / "states.pkl", "rb") as f:
    states = pickle.load(f)

# Identify the number of qubits
N = int(jnp.round(jnp.log2(states.shape[0])))

########################################################
# Define the contraction string for the tensor product #
########################################################

# Contra-variant indices
contravariant_idx = string.ascii_uppercase[:N]

# Co-variant indices
covariant_idx = string.ascii_lowercase[:N]

# Interleaved pairs
inter_pairs = [f"{A}{a}" for A, a in zip(contravariant_idx, covariant_idx)]

# Input string
input_str = ",".join(inter_pairs)

# Output string
output_str = "".join(inter_pairs)

# Contraction string
tensor_string = f"{input_str}->{output_str}"

#################################################
# Define the contraction string for the average #
#################################################

# Contra-variant indices
dual_idx = string.ascii_uppercase[:N]

# Co-variant indices
vector_idx = string.ascii_lowercase[:N]

# Interleaved pairs
inter_pairs = [f"{A}{a}" for A, a in zip(dual_idx, vector_idx)]

# Input string of the operator
input_str = "".join(inter_pairs)

# Contraction string
average_string = f"{dual_idx},{input_str},{vector_idx}->{''}"

########################################################

# Define the single-qubit Pauli matrices
I = jnp.array([[1., 0.], [0., 1.]], dtype=jnp.complex128)
X = jnp.array([[0., 1.], [1., 0.]], dtype=jnp.complex128)
Y = jnp.array([[0., -1j], [1j, 0.]], dtype=jnp.complex128)
Z = jnp.array([[1., 0.], [0., -1.]], dtype=jnp.complex128)

# Store the four Pauli matrices in a list
single_pauli = [I, X, Y, Z]

# Generate all possible combinations of N Pauli matrices
pauli_list = list(it.product(single_pauli, repeat=N))

# Initialise the list to store the curated Pauli strings
pauli_list_curated = []

# Iterate over all the tuples of Pauli matrices
for pauli_tuple in pauli_list:

    # Count the number of X or Y
    number_XY = sum(((p is X) or (p is Y)) for p in pauli_tuple)

    # Choose Pauli strings with even number of X or Y. 
    # (The expectatation value of Pauli strings with odd number of X or Y is zero in states with definite Hamming weight.)
    if number_XY % 2 == 0:

        # Compute the Pauli string with coefficient one
        tensor_product = jnp.einsum(tensor_string, *pauli_tuple)

        # Store the Pauli string
        pauli_list_curated.append(tensor_product)

def stabiliser_entropy(states, pauli_set, average_string, threshold=1e-10):
    """
    Computes the second stabiliser Rényientropy for each state |ψₐ⟩ in an array of D states,
    each of size 2**N:

        S₂(ψₐ) = N − log₂ ( ∑_P ⟨ψₐ|P|ψₐ⟩⁴ ) ,

    and prints also the identity test

        ∑_P ⟨ψₐ|P|ψₐ⟩² / 2ⁿ  = 1 .

    Parameters
    ----------
    states : jnp.ndarray
        Array of shape (2**N, D), whose columns are normalised pure states |ψₐ⟩.

    pauli_set : list of jnp.ndarray
        List of N-qubit Pauli strings P.

    average_string : string
        Einsum contraction string of the form 'AB...,AaBb...,ab...->', used to compute ⟨ψ|P|ψ⟩.

    threshold : float, optional
        Numerical resolution below which S₂ ≈ 0 is interpreted as 
        “stabiliser state”.
        Default: 1e−10.

    Returns
    -------
    list
        List [S₂(ψ₀), S₂(ψ₁), …, S₂(ψ_{D−1})], ordered according to the input states.
    """

    # Dimensions
    A, D = states.shape

    # Number of qubits
    N = int(jnp.round(jnp.log2(A)))

    if N < 10:
        header = "x"*22
    else:
        header = "x"*23

    print(header)
    print(f"Number of qubits N = {N}")
    print(header + "\n")

    # Container for the returned stabiliser entropies
    stabiliser_entropies = []

    # Iterate over states
    for a in range(D):

        # Extract |ψ⟩ and reshape to (2,)*N
        psi = states[:, a].reshape((2,)*N)

        # Initialise accumulators
        sum_exp_values_2 = 0.0
        sum_exp_values_4 = 0.0

        # Loop over Pauli strings
        for P in pauli_set:

            # Expectation ⟨ψ|P|ψ⟩
            exp_val = jnp.einsum(average_string, jnp.conj(psi), P, psi)

            # Contribution to ∑ |⟨ψ|P|ψ⟩|² / 2ⁿ
            sum_exp_values_2 += (exp_val**2) / (2**N)

            # Contribution to ∑ |⟨ψ|P|ψ⟩|⁴
            sum_exp_values_4 += exp_val**4

        # second stabiliser Rényientropy
        S = N - jnp.log2(sum_exp_values_4)

        # Store the entropy for return
        stabiliser_entropies.append(float(S.real))

        print("Test of the identity  ∑_P ⟨ψ|P|ψ⟩² / 2ⁿ  = 1.\n")
        print(f"{sum_exp_values_2.real:.18f}\n")

        print("Second stabiliser Rényi entropy S_2 = N − log₂ ∑_P ⟨ψ|P|ψ⟩⁴.\n")
        print(f"{S.real:.18f}\n")

        # Classification
        if S <= threshold:
            print(f"The state |ψ_{a}⟩ is a stabiliser state "
                  f"within numerical resolution {threshold}.\n")
        else:
            print(f"The state |ψ_{a}⟩ is not a stabiliser state "
                  f"within numerical resolution {threshold}.\n")

        print("-"*45 + "\n")

    # Return list of stabiliser entropies
    return stabiliser_entropies

def extremal_stabiliser_entropy(stabiliser_entropies, mode_numbers, N, percentage=0.05):
    """
    Identify, 
    from a list of second stabiliser Rényi entropies S₂[ψ_a] associated with a family of Bethe states,
    the fraction of states exhibiting extremal stabiliser entropy.

    The function assumes that

        stabiliser_entropies = [S₂[ψ_0], …, S₂[ψ_{D-1}]]

    and that `mode_numbers` is an array of shape (M, D), whose columns list the
    mode numbers labelling the corresponding Bethe state. Column indices are
    assumed to match those of `stabiliser_entropies`.

    Parameters
    ----------
    stabiliser_entropies : array-like
        One-dimensional list or array of length D containing the stabiliser Rényi-2
        entropies S₂[ψ_a].

    mode_numbers : array-like
        A jax.numpy or numpy array of shape (M, D), where each column stores the
        mode numbers indexing the corresponding Bethe state.

    N : int
        Total number of qubits, including those at the boundaries.

    percentage : float, optional
        Fraction of states to extract at the top and bottom of the stabiliser-entropy
        distribution. Defaults to 0.05.

    Returns
    -------
    None
    """

    # Convert mode numbers and entropies to JAX arrays
    mode_numbers = jnp.array(mode_numbers, dtype=jnp.int32)

    stabiliser_entropies = jnp.array(
        stabiliser_entropies,
        dtype=jnp.float64
    )

    # Number of Bethe states D and Hamming weight M
    M, D_check = mode_numbers.shape

    D = stabiliser_entropies.shape[0]

    # Check consistency between labels and entropy list
    if int(D_check) != int(D):
        raise ValueError(
            f"Inconsistent data: stabiliser-entropy list contains D = {D} states, "
            f"whereas the array of mode numbers contains D = {D_check} states."
        )

    # Number of states to extract
    count = max(1, int(percentage * D))

    # Host copy for sorting
    stabiliser_host = jnp.array(stabiliser_entropies, dtype=jnp.float64)
    stabiliser_host = jnp.asarray(stabiliser_host)
    stabiliser_host = jnp.array(stabiliser_host, dtype=jnp.float64)
    stabiliser_host = jnp.array(stabiliser_host, dtype=jnp.float64)

    stabiliser_host_np = jnp.array(stabiliser_host, dtype=float)

    # Indices ordered by increasing S₂
    sorted_indices = np.argsort(stabiliser_host_np)

    # Lowest and highest sectors
    lowest_indices  = sorted_indices[:count]
    highest_indices = sorted_indices[-count:]

    # Number of qubits in the bulk
    N_bulk = N - 2

    # Determine header width according to the number of digits in N−2 and M
    digits_N = len(str(N_bulk))
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
    print(f"N = {N_bulk} qubits in the bulk, Hamming weight M = {M}, and no domain walls.")
    print(header_bottom)

    if M == 1:
        print(f"The admissible mode numbers are 1 ≤ n ≤ {N_bulk - M + 1}" + "\n")

    elif M == 2:
        print(f"The admissible mode numbers are 1 ≤ n_1 < n_2 ≤ {N_bulk - M + 1}" + "\n")

    else:
        print(f"The admissible mode numbers are 1 ≤ n_1 < ... < n_{M} ≤ {N_bulk - M + 1}" + "\n")

    # Header: highest stabiliser entropies
    print("x"*56)
    print("Bethe states of highest second stabiliser Rényi entropy.")
    print("x"*56 + "\n")

    for idx in highest_indices:

        # Mode numbers of the state
        mode_nums = tuple(int(x) for x in mode_numbers[:, idx])

        # Stabiliser entropy of the state
        S2_val = float(stabiliser_host_np[idx])

        # Pretty-print
        mode_nums_str = ", ".join(str(q) for q in mode_nums)

        print(
            f"The Bethe state labelled by mode numbers {mode_nums_str} "
            f"has second stabiliser Rényi entropy "
            f"S_2 = {S2_val:.6f}"
        )

        print()

    # Header: lowest stabiliser entropies
    print("x"*55)
    print("Bethe states of lowest second stabiliser Rényi entropy.")
    print("x"*55 + "\n")

    for idx in lowest_indices:

        mode_nums = tuple(int(x) for x in mode_numbers[:, idx])
        S2_val = float(stabiliser_host_np[idx])

        mode_nums_str = ", ".join(str(q) for q in mode_nums)

        print(
            f"The Bethe state labelled by mode numbers {mode_nums_str} "
            f"has second stabiliser Rényi entropy "
            f"S_2 = {S2_val:.6f}"
        )

        print()

    return

stabiliser_entropies = stabiliser_entropy(states, pauli_list_curated, average_string)

# Load the data
with open(data_dir / "mode_numbers.pkl", "rb") as f:
    mode_numbers = pickle.load(f)

# Compute the fraction of Bethe states inside the eigenspace with extremal second stabiliser Rényi entropy, 
# if present

extremal_stabiliser_entropy(stabiliser_entropies, mode_numbers, N)