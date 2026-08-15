# Python 3.10.18

from jax import config as jax_config
jax_config.update("jax_platforms", "cpu")  

import jax
jax.config.update("jax_enable_x64", True)
print(jax.default_backend(), jax.devices())

import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import scipy.sparse as sp

from math import comb
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

# Define the number of qubits
N = 10

# Define the Hamming weight of the eigenspace with no domain walls
M = 4

# Define sparse Pauli matrices
I = sp.eye(2, dtype=np.float64, format='csr')
X = sp.csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
Y = sp.csr_matrix([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = sp.csr_matrix([[1, 0], [0, -1]], dtype=np.float64)

def embed_matrix(M, j, N):
    """
    Canonically promotes a sparse 2 x 2-matrix M on one qubit to a sparse 2**N x 2**N-matrix M_j on the j-th qubit of N qubits; 
    in particular, 
    it defines
        
    M_j = I ⊗ … ⊗ I ⊗ M ⊗ I ⊗ … ⊗ I,

    where M is the j-th factor (0 ≤ j ≤ N - 1).

    Parameters
    ----------
    matrix : sp.csr_matrix
        Sparse 2 x 2-matrix, e.g. I, X, Y, or Z.
    j : int
        Target qubit index.
    N : int
        Total number of qubits.

    Returns
    -------
    sp.csr_matrix
        2**N x 2**N-matrix operator.
    """
    # Verify index range
    if not (0 <= j < N):
        raise ValueError("Index j must satisfy 0 ≤ j ≤ N−1.")

    # Initialisation of the Kronecker product as a scalar
    op = sp.csr_matrix([[1.0]])

    # Sparse Kronecker product
    for k in range(N):
        op = sp.kron(op, M if k == j else I, format='csr')

    return op

# Define the lists of embedded sparse Pauli matrices over N + 2 qubits
I_list = [embed_matrix(I, j, N + 2) for j in range(N + 2)]
X_list = [embed_matrix(X, j, N + 2) for j in range(N + 2)]
Y_list = [embed_matrix(Y, j, N + 2) for j in range(N + 2)]
Z_list = [embed_matrix(Z, j, N + 2) for j in range(N + 2)]

def hamiltonian(N):
    """
    Define the sparse Hamiltonian over N + 2 qubits of the folded spin-1/2 XXZ model with open boundaries, 
    namely

        H = −(1/4) Σ_j (1 + Z_j Z_{j + 3}) x (X_{j + 1} X_{j + 2} + Y_{j + 1} Y_{j + 2}).

    Parameters
    ----------
    N : int
        Number of qubits.

    Returns
    -------
    sp.csr_matrix
        2**(N + 2) × 2**(N + 2) sparse Hamiltonian.
    """
    # Raise an error if N + 2 is less than four
    if N < 2:
        raise ValueError(f"Received N + 2 = {N + 2} < 4.")

    # Initialise the sparse matrix as the zero 2**(N + 2) x 2**(N + 2)-matrix
    D = 2 ** (N + 2)
    H = sp.csr_matrix((D, D), dtype=np.float64)

    # Define the Hamiltonian
    for j in range(N - 1):
        H -= 0.25 * ( I_list[j] @ I_list[j + 3] + Z_list[j] @ Z_list[j + 3] ) @ ( X_list[j + 1] @ X_list[j + 2] + Y_list[j + 1] @ Y_list[j + 2] )

    return H

def projector(N, M):
    """
    Define the sparse projector onto the eigenspace of the Hamiltonian over N + 2 qubits with Hamming weight M and no domain walls.

    Parameters
    ----------
    N : int
        Number of bulk qubits, which together with the boundaries sum up to N + 2.

    M : int
        Hamming weight.

    Returns
    -------
    scipy.sparse.csr_matrix
        2**(N + 2) × binomial(N - M + 1, M)-sparse matrix whose columns are the states of the computational basis
        |i_0 i_1 i_2 ... i_N i_{N+1}⟩ fulfilling simultaneously:
            i.-   i_0 = i_{N+1} = 0,
            ii.-  sum_{j=1}^N i_j = M,
            iii.- sum_{j=0}^{N} |i_{j+1} - i_j| = 2M.
    """
    # Initialiase the list storing the bit strings fulfilling the constraints.
    bit_strings = []

    # Iterate over bit strings
    for internal in range(2**N):
        bits = f"{internal:0{N}b}"

        # Hamming weight M.
        if bits.count('1') != M:
            continue

        # Ones must be isolated.
        if '11' in bits:
            continue

        # Insert extremal zeros.
        full_bits = "0" + bits + "0"

        # Verify the domain-wall condition.
        L = sum(abs(int(full_bits[j+1]) - int(full_bits[j])) for j in range(N + 1))
        if L != 2*M:
            continue

        # Represent the bit strings as integers.
        bit_strings.append(int(full_bits, 2))

    # Identity columns corresponding to the valid bit strings.
    cols = np.arange(len(bit_strings))
    data = np.ones(len(bit_strings))

    # Sparse projector.
    P = sp.csr_matrix((data, (bit_strings, cols)), shape=(2**(N + 2), len(bit_strings)))

    # Verification.
    expected_rows = 2**(N + 2)
    expected_cols = comb(N - M + 1, M)

    if P.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"The projector is a {P.shape[0]} x {P.shape[1]}-matrix, "
            f"whilst it must be a {expected_rows} x {expected_cols})-matrix."
        )

    return P

def plot_eigenvalues(N, M, D):
    """
    Compute the k smallest eigenvalues of the projection of the Hamiltonian onto the subspace of Hamming weight M with no domain walls.

    Parameters
    ----------
    N : int
        Number of bulk qubits, which together with the boundaries sum up to N + 2.

    M : int
        Hamming weight.

    D : int
        Number of eigenvalues to extract.

    Returns
    -------
    np.ndarray
        Array of the k smallest eigenvalues of the projection of the Hamiltonian onto the subspace of Hamming weight M with no domain walls,
        sorted in ascending order.
    """
    # Hamiltonian
    H = hamiltonian(N)

    # Projector
    P = projector(N, M)

    # Projected Hamiltonian
    HM = P.T @ (H @ P)

    # Sparse extraction of the k smallest eigenvalues
    evals = sp.linalg.eigsh(
        HM,
        k=D,
        which='SM',
        return_eigenvectors=False
    )

    # Sort into real 
    evals = np.sort(np.real(evals))

    # Discrete eigenvalue plot
    plt.figure(figsize=(7, 4))
    plt.scatter(range(1, D + 1), evals, s=20)  
    plt.xlabel("Position of the eigenvalues", fontsize=11)
    plt.ylabel("Energy ${E}$", fontsize=11)
    plt.title(
        rf"$N={N}$ qubits and Hamming weight $M={M}$ (no DWs).",
        fontsize=12
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return evals

def eigenstates(N, M, D, sigma=1e-10):
    """
    Extract the D eigenstates of the Hamiltonian projected onto the subspace of Hamming weight M with no domain walls whose eigenvalues lie closest to the centre of the spectrum.

    Parameters
    ----------
    N : int
        Number of bulk qubits, which together with the boundaries sum up to N + 2.

    M : int
        Hamming weight of the constrained subspace.

    D : int
        Number of central eigenstates to extract.

    sigma : float, optional
        Shift-invert target. sigma = 1e-10 targets the centre of the spectrum.

    Returns
    -------
    states : np.ndarray
        2**(N+2) × D-matrix, whose columns are the normalised eigenvectors |Ψ_a⟩.
    """

    # Hamiltonian on N+2 qubits
    H = hamiltonian(N)

    # Projector onto constrained subspace
    P = projector(N, M)

    # Projected Hamiltonian
    H_M = P.T @ (H @ P)

    # Dimension of the subspace
    D_M = H_M.shape[0]

    if D > D_M:
        raise ValueError(
            "Cannot extract more eigenstates than the dimension of the subspace."
        )

    # Sparse diagonalisation, shift–inverted and centred around zero.
    evals, evecs = sp.linalg.eigsh(
        H_M,
        k=D,
        sigma=sigma,
        which="LM",
        return_eigenvectors=True
    )

    # Sort eigenstates by ascending eigenvalue.
    order = np.argsort(evals)
    evecs = evecs[:, order]       # shape (D_M, D)

    # Lift the eigenstates to the full Hilbert space: |Ψ_a⟩ = P |ψ_a⟩.
    states = P @ evecs            # shape (2^(N+2), D)
    states = np.array(states, dtype=float)

    # Normalise each eigenstate
    for j in range(D):
        v = states[:, j]
        states[:, j] = v / np.linalg.norm(v)

    #Convert to jnp.array
    states = jnp.array(states)

    return states

# Number of eigenvalues
D = comb(N - M + 1, M) - 2

# Plot eigenvalue spectrum for this sector
plot_eigenvalues(N, M, D)

# Compute as many eigenstates
states = eigenstates(N, M, D)
  
if "__file__" in globals():
    script_dir = Path(__file__).resolve().parent
else:
    script_dir = Path.cwd()

# Define the target directory 
states_dir = script_dir.parent / "data"

# Ensure the directory exists
states_dir.mkdir(parents=True, exist_ok=True)

# Define the full path of the .pkl
pkl_filename = states_dir / "states.pkl"

# Save the array of eigenstates
with open(pkl_filename, "wb") as f:
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

# Confirmation message
print(f"Eigenstates over N + 2 = {N + 2} qubits, with Hamming weight M = {M} and no domain walls, saved successfully at {pkl_filename}.")