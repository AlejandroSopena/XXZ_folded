import os
import pickle
import subprocess
import time

import argparse
import numpy as np
import cupy as cp
from scipy.optimize import curve_fit
from qiskit import transpile, qasm2

from qibo import set_backend, set_precision, set_threads, gates, Circuit
from qibo.quantum_info import fidelity
from qibo.noise import NoiseModel, DepolarizingError
from qibo.models.error_mitigation import sample_training_circuit_cdr
from qibo.backends import _check_backend_and_local_state, construct_backend
from qibo.symbols import I
from qibo.hamiltonians import SymbolicHamiltonian

from XXZ_folded import XXZ_folded_one_domain

def density_matrix_to_state_vector(rho):
    # Check if the density matrix represents a pure state by verifying Tr(rho^2) = 1
    if np.isclose(np.trace(rho @ rho), 1.0):
        # Perform eigendecomposition to get the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        
        # The pure state corresponds to the eigenvector with eigenvalue close to 1
        pure_state_index = np.argmax(eigenvalues)
        pure_state_vector = eigenvectors[:, pure_state_index]
        
        return pure_state_vector
    else:
        raise ValueError("The density matrix does not represent a pure state.")
    
def main():
    parser = argparse.ArgumentParser(description="Run simulation with specified parameters.")
    parser.add_argument('--basis_gates', nargs='+', default=['cx', 'rz', 'sx', 'x', 'id'], help='List of basis gates')
    parser.add_argument('--boundaries', type=bool, default=False, help='Boundaries flag')
    parser.add_argument('--lamb', type=float, default=3e-3, help='Lambda value')
    parser.add_argument('--n_training_samples', type=int, default=50, help='Number of training samples')
    parser.add_argument('--path', type=str, default='result', help='Path to save states')
    parser.add_argument('--N', type=int, default=9, help='Number of qubits')
    parser.add_argument('--M', type=int, default=1, help='Number of magnons')
    parser.add_argument('--D', type=float, default=4, help='Number of domain walls')
    parser.add_argument('--domain_pos', nargs='+', type=int, default=[[4,5],[8,9]], help='Domain positions') # [[3,4],[7,8,9],[12,13]]
    parser.add_argument('--connectivity', type=str, default=None, help='Connectivity type')
    parser.add_argument('--precision', type=str, default='double', help='Precision type')

    args = parser.parse_args()

    basis_gates = args.basis_gates
    boundaries = args.boundaries
    lamb = args.lamb
    n_training_samples = args.n_training_samples
    path = args.path
    N = args.N
    M = args.M
    D = args.D
    domain_pos = args.domain_pos
    connectivity = args.connectivity
    precision = args.precision

    if lamb == 0:
        noise_model = None
    else:
        noise_model = NoiseModel() 
        noise_model.add(DepolarizingError(lamb),gates.CNOT)


    if connectivity == 'google_sycamore':
        if N == 5:
            connectivity = np.load('connectivities/connectivity_google_sycamore_12.npy',allow_pickle=True).tolist()
        elif N == 6:
            connectivity = np.load('connectivities/connectivity_google_sycamore_13.npy',allow_pickle=True).tolist()
        else:
            connectivity = None

    coupling_map = connectivity

    if noise_model is None:
        density_matrix = False
    else:
        density_matrix = True

    os.makedirs(path, exist_ok=True)
    os.makedirs(path+'/training_states', exist_ok=True)

    set_backend("qibojit", platform="numba")
    set_threads(20)

    model = XXZ_folded_one_domain(N, M, D, domain_pos)
    model._get_roots()
    circ_xx, circ_xxb = model.get_xx_b_circuit()
    circ_u0 = model.get_U0_circ()
    set_precision('single')
    circ_d = model.get_D_circ()
    circ_Psi_M_0 = model.get_Psi_M_0_circ()

    circ = model.get_full_circ()

    circ_qiskit = model.circ_to_qiskit(circ)

    circ_qiskit1 = transpile(circ_qiskit,basis_gates=basis_gates,coupling_map=coupling_map,optimization_level=3,layout_method='sabre',routing_method='sabre')

    layout_final = []
    for q in circ_qiskit1.layout.final_layout.get_virtual_bits().values():
        layout_final.append(q)
    print(layout_final)

    qasm_code = qasm2.dumps(circ_qiskit1)
    circ = Circuit.from_qasm(qasm_code)

    print(circ.gate_types)
    print(circ.depth)
    print(circ.nqubits)

    #model.circ_full = circ

    set_backend("qibojit",platform="numba")
    set_precision(precision)
    backend = construct_backend("qibojit",platform="numba")
    backend.set_precision(precision)

    layout_final = None
    state_noiseless = model.get_state(density_matrix=False, boundaries=boundaries, layout=layout_final, backend=backend)
    # s = cp.asnumpy(state_noiseless)
    # s = density_matrix_to_state_vector(s)
    # s = cp.array(s)
    # print(np.where(abs(s)>1e-10))
    #print(np.where(abs(np.linalg.eigvals(s))>1e-10))
    energy_noiseless = model.get_energy(state_noiseless, boundaries=boundaries)
    Q1_noiseless = model.get_magnetization(state_noiseless, boundaries=boundaries)
    Q2_noiseless = model.get_correlation(state_noiseless, boundaries=boundaries)

    if backend.platform == 'cupy':
        energy_noiseless = float(energy_noiseless.get())
        Q1_noiseless = float(Q1_noiseless.get())
        Q2_noiseless = float(Q2_noiseless.get())
        
    fid_ham, fid_q1, fid_q2 = model.check_fidelity(state_noiseless, boundaries=boundaries)
    print("  Fidelity Hamiltonian: ", fid_ham)
    print("  Fidelity Q1: ", fid_q1)
    print("  Fidelity Q2: ", fid_q2)

    print('Noiseless')
    print("  Energy: ", energy_noiseless)
    print("  Q1: ", Q1_noiseless)
    print("  Q2: ", Q2_noiseless)

    start_time = time.time()
    state_noise = model.get_state(density_matrix=density_matrix, boundaries=boundaries, noise_model=noise_model, layout=layout_final, backend=backend)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    cp.get_default_memory_pool().free_all_blocks()

    fid = fidelity(state_noiseless, state_noise)

    energy = model.get_energy(state_noise, boundaries=boundaries)
    q1_val = model.get_magnetization(state_noise, boundaries=boundaries)
    q2_val = model.get_correlation(state_noise, boundaries=boundaries)

    print('Noisy')
    print("  Fidelity: ", fid)
    print("  Energy:", energy)
    print("  Q1: ", q1_val)
    print("  Q2: ", q2_val)

    np.save(path+'/state.npy', {'noiseless': state_noiseless, 'noisy': state_noise})

    del state_noiseless
    del state_noise

    cp.get_default_memory_pool().free_all_blocks()

    circ = model.circ_full

    seed = None
    backend = None
    backend, local_state = _check_backend_and_local_state(seed, backend)

    training_circuits = [
        sample_training_circuit_cdr(circ, seed=local_state, backend=backend)
        for _ in range(n_training_samples)
    ]

    np.save(path+'/training_states/training_circuits.npy', training_circuits)

    for i in range(n_training_samples):
        print('Training circuit: ', i)
        with open('input_args.pkl', "wb") as f:
            pickle.dump((model, i, path, noise_model, boundaries, density_matrix, layout_final, backend), f)
        subprocess.run(["python", "training_states.py"])

    states = np.load(path+'/state.npy',allow_pickle=True).item()
    noisy_state = states['noisy']

    hamiltonian = model.get_xxz_folded_hamiltonian(boundaries)
    q1 = model.get_q1(boundaries) - (model.N/2)*SymbolicHamiltonian(I(model.N-1))
    q2 = model.get_q2(boundaries) - ((model.N+1)/2)*SymbolicHamiltonian(I(model.N-1))

    def get_mit_value(observable, n_training_samples, noisy_state):
        train_val = {"noiseless": [], "noisy": []}
        for i in range(n_training_samples):
            training_state = np.load(path+f'/training_states/states_{i}.npy',allow_pickle=True).item()
            state = training_state['noiseless']
            
            val = observable.expectation(state)
            if backend.platform == 'cupy':
                val = float(val.get())
            train_val["noiseless"].append(val)

            state = training_state['noisy']
            val = observable.expectation(state)
            if backend.platform == 'cupy':
                val = float(val.get())
            train_val["noisy"].append(val)

        nparams = 2 

        params = local_state.random(nparams)
        f = lambda x, a, b: a * x + b

        optimal_params = curve_fit(
            f,
            train_val["noisy"],
            train_val["noise-free"],
            p0 = params,
        )[0]

        val = observable.expectation(noisy_state)
        if backend.platform == 'cupy':
                val = float(val.get())
        mit_val = f(val, *optimal_params)

        return mit_val, val, optimal_params, train_val

    observables = [hamiltonian, q1, q2]
    observables_label = ['Energy', 'Q1', 'Q2']

    results = [[[circ,layout_final],energy_noiseless,Q1_noiseless,Q2_noiseless]]
    for i, observable in enumerate(observables):
        mit_val, val, optimal_params, train_val = get_mit_value(observable, n_training_samples, noisy_state)
        results.append([mit_val, val, optimal_params, train_val])
        print(observables_label[i])
        print("  Mitigated: ", mit_val)
        print("  Noisy: ", val)
        print("  Optimal parameters: ", optimal_params)

    results = np.array(results, object)

    np.save(path+'/mitigated_values.npy', results, allow_pickle=True)

if __name__ == "__main__":
    main()
