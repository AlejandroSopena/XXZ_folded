# Python 3.10.18
from jax import config as jax_config
jax_config.update("jax_platforms", "cpu")  # set before importing jax

import jax
jax.config.update("jax_enable_x64", True)
print(jax.default_backend(), jax.devices())

import jax.numpy as jnp
import pickle
import numpy as np
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

########################################################
# Define the contraction string for the matrix product #
########################################################

# Define the Greek alphabet
greek_alphabet = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"
]

# Contra-variant indices
contravariant_idx = string.ascii_uppercase[:N]

# Contracted indices
contracted_idx = string.ascii_lowercase[:N]

# Covariant indices
covariant_idx = greek_alphabet[:N]

# Interleaved pairs
first_inter_pairs = [f"{A}{a}" for A, a in zip(contravariant_idx, contracted_idx)]
second_inter_pairs = [f"{a}{g}" for a, g in zip(contracted_idx, covariant_idx)]
output_inter_pairs = [f"{A}{g}" for A, g in zip(contravariant_idx, covariant_idx)]

# Input string of the operator
first_input_str = "".join(first_inter_pairs)
second_input_str = "".join(second_inter_pairs)
output_str = "".join(output_inter_pairs)

# Contraction string
matrix_string = f"{ first_input_str},{second_input_str}->{output_str}"

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
I = jnp.array([[1., 0.], [0., 1.]], dtype=jnp.float64)
X = jnp.array([[0., 1.], [1., 0.]], dtype=jnp.float64)
Y = jnp.array([[0., -1j], [1j, 0.]], dtype=jnp.complex128)
Z = jnp.array([[1., 0.], [0., -1.]], dtype=jnp.float64)

def majorana(N, tensor_string):
    """
    Construct the full set of 2N Majorana operators {γ_0, ..., γ_{2N−1}} over N qubits.

    The definition of the Majorana operators is

        γ_{2j}   = (∏_{k=0}^{j−1} Z_k) ⊗ X_j ⊗ (∏_{l=j+1}^{N−1} I_l)
        γ_{2j+1} = (∏_{k=0}^{j−1} Z_k) ⊗ Y_j ⊗ (∏_{l=j+1}^{N−1} I_l)

    Majorana operators satisfy the canonical anti‑commutation relations

        {γ_j, γ_k} = 2δ_{jk} · I

    Parameters
    ----------
    N : int
        Number of qubits.

    tensor_string : dict
        Einsum contraction string of the form "Aa,Bb,...->AaBb...".

    Returns
    -------
    gammas : dict
        Dictionary mapping integer labels k = 0, ..., 2N−1 to the corresponding Majorana operator γ_k, represented as a jnp.ndarray.
    """
    # Initialise the dictionary to store Majorana operators
    gammas = {}
    
    # Build γ_{2k} and γ_{2k+1} for each site k
    for k in range(N):
        
        # List of Pauli matrices for γ_{2k}
        Pauli_even = [Z]*k + [X] + [I]*(N - k - 1)
        
        # List of Pauli matrices for γ_{2k+1}
        Pauli_odd = [Z]*k + [Y] + [I]*(N - k - 1)
        
        # Compute Majorana operators via einsum
        gammas[2*k] = jnp.einsum(tensor_string, *Pauli_even)
        gammas[2*k+1] = jnp.einsum(tensor_string, *Pauli_odd)
    
    # Return the dictionary of Majorana operators
    return gammas

gammas = majorana(N, tensor_string)

def covariance_matrix(states, gammas, matrix_string, average_string):
    """
    Computes the covariance matrices, whose matrix elements are
    
        C_a,jk = -(i/2) ⟨ψ_a| [γ_j, γ_k] |ψ_a⟩.

    Parameters
    ----------
    states : jnp.ndarray
        Array of shape (2**N, D), 
        whose columns are normalised states |ψ⟩.
        Each column is internally reshaped as a tensor (2,)*N prior to contraction.

    gammas : dict
        Dictionary mapping integer labels k = 0, ..., 2N−1 to the corresponding Majorana operator γ_k, 
        represented as a jnp.ndarray.

    matrix_string : string
        Einsum contraction string of the form 'AaBb...,aαbβ...->AαBβ...', 
        used to compute γ_j γ_k by tensor contraction.

    average_string : string
        Einsum contraction string of the form 'AB...,AaBb...,ab...->', 
        used to C_jk by tensor contraction.

    Returns
    -------
    list of jnp.ndarray
        A list [C_0, …, C_D−1],
        where each C_a is the skew-symmetric covariance 2N × 2N-matrix of the a-th state.
    """
    # Number of components and number of states
    A, D = states.shape

    # Number of qubits
    N = int(jnp.round(jnp.log2(A)))

    # Initialise the list of covariance matrices
    C_list = []

    # Loop over each state vector
    for a in range(D):

        # Extract the a-th state and frame it as a N-times contravariant tensor
        psi = states[:, a]
        psi = psi.reshape((2,) * N)

        # Initialise the covariance 2N × 2N-matrix
        C = jnp.zeros((2*N, 2*N), dtype=jnp.complex128)

        # Loop over first index of the Majorana operator
        for j in range(2*N):

            # Loop over second index of the Majorana operator
            for k in range(2*N):

                # Diagonal entries vanish identically
                if j == k:
                    continue

                # Fetch γ_j and γ_k
                ga = gammas[j]
                gb = gammas[k]

                # Compute γ_j γ_k
                gagb = jnp.einsum(matrix_string, ga, gb)

                # Compute ⟨ψ_a| γ_j γ_k |ψ_a⟩
                exp_val = jnp.einsum(average_string, jnp.conj(psi), gagb, psi)

                # Append the entry to the covariant matrix
                C = C.at[j, k].set(-1j * exp_val)

        # Enforce skew-symmetry
        C = (C - C.T) / 2

        # Append the covariance matrix for the a-th state
        C_list.append(C)

    return C_list

covariance_states = covariance_matrix(states, gammas, matrix_string, average_string)

def antiflatness(C_list, threshold=1e-10):
    """
    Compute the fermionic anti-flatness of each covariance matrix C in the list.

        A = N − (1/2) Tr(Cᵀ C)

    If A = 0, 
    the corresponding state |ψ⟩ is fermionically Gaussian; 
    if A > 0, 
    the state is fermionically non-Gaussian.

    Parameters
    ----------
    C_list : list of jnp.ndarray
        List [C_0, …, C_{D−1}], 
        where each C_a is a skew-symmetric covariance 2N × 2N-matrix.

    threshold : float, optional
        Numerical threshold below which A is interpreted as zero.
        Default: 1e−10.

    Returns
    -------
    anti_flatness_values : list of floats
        List [A_0, …, A_{D−1}] containing the fermionic anti-flatness of each covariance matrix,
        in the same order as provided in `C_list`.
    """

    # Number of covariance matrices
    D = len(C_list)

    # Empty input
    if D == 0:
        print("Empty list of covariance matrices.\n")
        return []

    # Verify each covariance matrix
    for idx, C in enumerate(C_list):

        if C.ndim != 2:
            raise ValueError(
                f"The array C_{idx} is not a matrix, but a tensor of {C.ndim} indices."
            )

        if C.shape[0] != C.shape[1]:
            raise ValueError(
                f"The covariance matrix C_{idx} is not square, but {C.shape[0]} × {C.shape[1]}."
            )

        if C.shape[0] % 2 != 0:
            raise ValueError(
                f"The covariance matrix C_{idx} has an odd dimension: {C.shape[0]} × {C.shape[0]}."
            )

    # Number of qubits
    N = C_list[0].shape[0] // 2

    # Header width
    width = 22 if N < 10 else 23
    header = "x" * width

    print(header)
    print(f"Number of qubits N = {N}")
    print(header + "\n")

    # Container for values
    anti_flatness_values = []

    # Loop over covariance matrices
    for a, C in enumerate(C_list):

        # Compute Cᵀ C
        CTC = C.T @ C

        # Anti-flatness A = N − (1/2) Tr(Cᵀ C)
        A = float(N - 0.5 * jnp.trace(CTC).real)
        anti_flatness_values.append(A)

        # Report
        print("Fermionic anti-flatness A = N − (1/2) Tr(Cᵀ C).\n")
        print(f"{A:.18f}\n")

        # Classification
        if A <= threshold:
            print(
                f"The state |ψ_{a}⟩ is fermionically Gaussian "
                f"within numerical resolution {threshold}.\n"
            )
        else:
            print(
                f"The state |ψ_{a}⟩ is fermionically non-Gaussian "
                f"within numerical resolution {threshold}.\n"
            )

        print("-" * 46 + "\n")

    return anti_flatness_values

def extremal_antiflatness(anti_flatness_values, mode_numbers, N, percentage=0.05):
    """
    Identify,
    from a list of fermionic anti-flatness values A[ψₐ] associated with a family of Bethe states,
    the fraction of states exhibiting extremal fermionic non-Gaussianity.

    The function assumes that

        anti_flatness_values = [A[ψ₀], …, A[ψ_{D−1}]],

    and that `mode_numbers` is an array of shape (M, D) whose columns list the
    mode numbers labelling the corresponding Bethe state.
    Column indices are assumed to match those of `anti_flatness_values`.

    Parameters
    ----------
    anti_flatness_values : array-like
        One-dimensional list or array of length D containing A[ψₐ]
        = N − (1/2) Tr(Cᵀ C) for each state.

    mode_numbers : array-like
        A jax.numpy or numpy array of shape (M, D), where each column stores the
        mode numbers indexing the corresponding Bethe state.

    N : int
        Total number of qubits (including boundaries).

    percentage : float, optional
        Fraction of states to extract at the top and bottom of the anti-flatness
        distribution. Defaults to 0.05.

    Returns
    -------
    None
    """

    # Convert inputs to JAX arrays
    mode_numbers = jnp.array(mode_numbers, dtype=jnp.int32)

    anti_flatness_values = jnp.array(
        anti_flatness_values,
        dtype=jnp.float64
    )

    # Dimensions
    M, D_check = mode_numbers.shape
    D = anti_flatness_values.shape[0]

    # Consistency check
    if int(D_check) != int(D):
        raise ValueError(
            f"Inconsistent data: anti-flatness list contains D = {D} states, "
            f"whereas the array of mode numbers contains D = {D_check} states."
        )

    # Number of states to extract
    count = max(1, int(percentage * D))

    # Host copy for sorting
    anti_host_np = np.array(anti_flatness_values, dtype=float)

    # Sorted indices
    sorted_indices = np.argsort(anti_host_np)

    # Lowest (closest to Gaussian) and highest (most non-Gaussian) states
    lowest_indices  = sorted_indices[:count]
    highest_indices = sorted_indices[-count:]

    # Bulk size
    N_bulk = N - 2

    # Header-width logic
    digits_N = len(str(N_bulk))
    digits_M = len(str(M))

    if digits_N == 1 and digits_M == 1:
        width = 68
    elif digits_N == 2 and digits_M == 2:
        width = 70
    else:
        width = 69

    header_top    = "\n" + "x" * width
    header_bottom = "x"  * width + "\n"

    # Global header
    print(header_top)
    print(f"N = {N_bulk} qubits in the bulk, Hamming weight M = {M}, and no domain walls.")
    print(header_bottom)

    # Admissible mode-number range
    if M == 1:
        print(f"The admissible mode numbers are 1 ≤ n ≤ {N_bulk - M + 1}\n")
    elif M == 2:
        print(f"The admissible mode numbers are 1 ≤ n₁ < n₂ ≤ {N_bulk - M + 1}\n")
    else:
        print(f"The admissible mode numbers are 1 ≤ n₁ < ... < n_{M} ≤ {N_bulk - M + 1}\n")

    # Header: highest anti-flatness (most non-Gaussian)
    print("x"*48)
    print("Bethe states of highest fermionic anti-flatness.")
    print("x"*48 + "\n")

    for idx in highest_indices:

        mode_nums = tuple(int(x) for x in mode_numbers[:, idx])
        A_val = float(anti_host_np[idx])

        mode_nums_str = ", ".join(str(q) for q in mode_nums)

        print(
            f"The Bethe state labelled by mode numbers {mode_nums_str} "
            f"has fermionic anti-flatness "
            f"A = {A_val:.6f}"
        )
        print()

    # Header: lowest anti-flatness (closest to Gaussian)
    print("x"*47)
    print("Bethe states of lowest fermionic anti-flatness.")
    print("x"*47 + "\n")

    for idx in lowest_indices:

        mode_nums = tuple(int(x) for x in mode_numbers[:, idx])
        A_val = float(anti_host_np[idx])

        mode_nums_str = ", ".join(str(q) for q in mode_nums)

        print(
            f"The Bethe state labelled by mode numbers {mode_nums_str} "
            f"has fermionic anti-flatness "
            f"A = {A_val:.6f}"
        )
        print()

    return

anti_flatnesses = antiflatness(covariance_states)

# Load the data
with open(data_dir / "mode_numbers.pkl", "rb") as f:
    mode_numbers = pickle.load(f)

# Compute the fraction of Bethe states inside the eigenspace with extremal second stabiliser Rényi entropy, 
# if present

extremal_antiflatness(anti_flatnesses, mode_numbers, N)