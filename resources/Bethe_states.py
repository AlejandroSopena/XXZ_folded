# Python 3.10.18

from jax import config as jax_config
jax_config.update("jax_platforms", "cpu")  

import jax
jax.config.update("jax_enable_x64", True)
print(jax.default_backend(), jax.devices())

import jax.numpy as jnp
import pickle

from itertools import combinations
from pathlib import Path

# Define the number of qubits
N = 9

# Define the Hamming weight of the eigenspace with no domain walls
M = 3

def momenta(N):
    """
    Computes all the quantised momenta compatible labelling domainless Bethe states of the folded spin–1/2 XXZ model with open boundaries.

        p_m = π m / (N + 1), m = 1, …, N .

    Parameters
    ----------
    N : int
        Number of qubits.

    Returns
    -------
    jnp.ndarray
        One–dimensional array of shape (N,) containing the allowed momenta p_m,
        represented in double precision (float64).
    """

    # Construct the sequence m = 1,...,N
    n = jnp.arange(1, N + 1, dtype=jnp.float64)

    # Denominator N + 1, promoted to float64
    denominator = jnp.float64(N + 1)

    # Quantised momenta p_m = π m / (N + 1)
    p = jnp.pi * n / denominator

    # Return the array of momenta
    return p

def momenta_and_modes(N, M):
    """
    Computes the vectorised set of momenta of the Hamiltonian of the spin-1/2 folded XXZ model with open boundaries over N + 2 qubits,
    projected onto the eigenspace of fixed Hamming weight M and no domain walls.

    For fixed N and M:
        • it determines the effective number of qubits N_eff = N − M + 1;
        • computes the complete set of admissible momenta p_m for m = 1, ..., N_eff;
        • enumerates all the combinations of M distinct integers in 1, ..., N_eff;
        • and extracts the corresponding subsets of momenta and in vectorised form.

    Parameters
    ----------
    N : int
        Number of qubits.

    M : int
        Hamming weight.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        A pair (ps, mode_numbers) where:
            • ps is a binomial(N_eff, M) x M-matrix whose rows contain the corresponding momentum subsets p_{m_1}, …, p_{m_M}.
            • mode_numbers is a binomial(N_eff, M) x M-matrix listing the mode numbers
    """

    # Effective number of qubits
    N_eff = N - M + 1

    # Complete list of admissible momenta.
    available_p = momenta(N_eff)                                                   

    # All combinations of distinct mode numbers
    mode_numbers = jnp.array(list(combinations(range(N_eff), M)), dtype=jnp.int32)

    # Vectorised extraction of momentum subsets
    ps = jnp.take(available_p, mode_numbers)                                       

    # Swap the roles of rows and columns for the momenta and mode numbers to be listed column-wise
    ps = ps.T                                                                      
    mode_numbers = mode_numbers.T                                                  

    # Return the pair of arrays
    return ps, mode_numbers + 1

def Bethe_states(N, p_subsets):
    """
    Computes the normalised Bethe states of the folded spin–1/2 XXZ chain with open boundaries over N + 2 qubits,
    in the eigenspace of fixed Hamming weight M and no domain walls.

    If the ordered configuration of spatial coordinates is

        1 ≤ n_1 < n_2 < ... < n_M ≤ N,

    and each subset of momenta p_{m_1}, ..., p_{m_M}, the Bethe wave function is

        ψ_a(n_1, ..., n_M) = det_{a,b} sin( (n_a − a + 1) p_{m_b} ).

    Parameters
    ----------
    N : int
        Number of bulk qubits. The full chain has N + 2 qubits.

    p_subsets : jnp.ndarray
        Momentum subsets of shape (M, K). Each column labels a distinct Bethe state.

    Returns
    -------
    jnp.ndarray
        2**(N+2) × K matrix whose columns are the normalised, boundary-embedded Bethe states.
    """
    
    # Hamming weight M and number of Bethe states K
    M, K = p_subsets.shape

    # Effective spatial coordinates in {1, ..., N_eff}, where N_eff = N − M + 1
    N_eff = N - M + 1
    n_eff = list(combinations(range(N_eff), M))
    n_eff = jnp.array(n_eff, dtype=jnp.int32) + 1 
    K_check = n_eff.shape[0]

    # Interrupt if the number of subsets of effective spatial coordinates disagrees with the number of subsets of momenta 
    if K_check != K:
        raise ValueError(
            f"Number of subsets of admissible spatial coordinates {K_check} differs from number of subsets of momenta {K}."
        )

    # Spatial coordinates n_a = n_a,eff + a − 1
    a_inds = jnp.arange(1, M+1).reshape(1, M)
    ns = n_eff + a_inds - 1

    # Integer index in Hilbert space of the bulk qubits: alpha = Σ_a 2^(N − n_a)
    powers = 2 ** (N - ns)
    bulk_indices = jnp.sum(powers, axis=1)

    # Slater matrices
    sin_matrix = jnp.sin(
        n_eff[:, :, None, None] *
        p_subsets[None, None, :, :]
    )
    sin_matrix = jnp.transpose(sin_matrix, (0, 3, 1, 2))

    # Slater determinants
    dets = jnp.linalg.det(sin_matrix)

    # Allocate Bethe states on the bulk Hilbert space 2**N
    dim_bulk = 2**N
    bethe_states = jnp.zeros((dim_bulk, K), dtype=jnp.float64)

    # Accumulate amplitudes
    for k in range(K):
        idx = bulk_indices[k]
        bethe_states = bethe_states.at[idx, :].add(dets[k, :])

    # Normalise the bulk Bethe states
    norms = jnp.linalg.norm(bethe_states, axis=0)
    bethe_states = bethe_states / norms

    # Boundary vector |0⟩
    zero = jnp.array([1.0, 0.0], dtype=jnp.float64)

    # Embedding |Ψ_a⟩ = |0⟩ ⊗ |ψ_a⟩ ⊗ |0⟩
    full_states = jnp.zeros((2**(N+2), K), dtype=jnp.float64)
    for k in range(K):
        psi_bulk = bethe_states[:, k]
        psi_full = jnp.kron(zero, jnp.kron(psi_bulk, zero))
        full_states = full_states.at[:, k].set(psi_full)

    return full_states

# Momenta and mode numbers
ps, ms = momenta_and_modes(N, M)

# Compute as many Bethe states
states = Bethe_states(N, ps)

if "__file__" in globals():
    script_dir = Path(__file__).resolve().parent
else:
    script_dir = Path.cwd()

# Define the target directory 
results_dir = script_dir.parent / "data"

# Ensure the directory exists
results_dir.mkdir(parents=True, exist_ok=True)

# Define the full path of the .pkl
states_filename = results_dir / "states.pkl"

# Save the array of eigenstates
with open(states_filename, "wb") as f:
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

# Define the full path of the .pkl
mode_filename = results_dir / "mode_numbers.pkl"

# Save the array of eigenstates
with open(mode_filename, "wb") as f:
    pickle.dump(ms, f, protocol=pickle.HIGHEST_PROTOCOL)

# Confirmation message
print(f"Bethe states over N + 2 = {N + 2} qubits, with Hamming weight M = {M} and no domain walls, saved successfully at {states_filename}; the associated mode numbers saved at {mode_filename}.")