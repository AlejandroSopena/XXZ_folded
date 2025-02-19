import os
import pickle
import subprocess
import time

import argparse
import numpy as np
from scipy.optimize import curve_fit
from qiskit import transpile, qasm2

from qibo import set_backend, set_precision, set_threads, gates, Circuit
from qibo.quantum_info import fidelity
from qibo.noise import NoiseModel, DepolarizingError
from qibo.models.error_mitigation import sample_training_circuit_cdr
from qibo.backends import _check_backend_and_local_state, construct_backend
from qibo.symbols import I
from qibo.hamiltonians import SymbolicHamiltonian

from XXZ_folded import XXZ_folded
    
def main():

    def parse_nested_list(s):
        try:
            return eval(s)
        except:
            raise argparse.ArgumentTypeError("Invalid format for nested list")
    
    parser = argparse.ArgumentParser(description="Run simulation with specified parameters.")
    parser.add_argument('--basis_gates', nargs='+', default=['cx', 'rz', 'sx', 'x', 'id'], help='List of basis gates')
    parser.add_argument('--boundaries', type=bool, default=False, help='Boundaries flag')
    parser.add_argument('--lamb', type=float, default=3e-3, help='Lambda value')
    parser.add_argument('--n_training_samples', type=int, default=50, help='Number of training samples')
    parser.add_argument('--path', type=str, default='result', help='Path to save states')
    parser.add_argument('--N', type=int, default=7, help='Number of qubits')
    parser.add_argument('--M', type=int, default=1, help='Number of magnons')
    parser.add_argument('--D', type=int, default=2, help='Number of domain walls')
    parser.add_argument('--domain_pos', type=parse_nested_list, default=[[5,6]], help='Domain positions')
    parser.add_argument('--connectivity', type=str, default=None, help='Connectivity type')
    parser.add_argument('--backend', type=str, default='numba', help='Calculation engine: numba or cupy')
    parser.add_argument('--precision', type=str, default='double', help='Precision type: double or single')
    parser.add_argument('--nthreads', type=int, default=8, help='Number of threads for numba')

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
    backend_name = args.backend
    precision = args.precision
    nthreads = args.nthreads

    if lamb == 0:
        noise_model = None
    else:
        noise_model = NoiseModel() 
        noise_model.add(DepolarizingError(lamb),gates.CNOT)


    if connectivity == 'google_sycamore':
        if N == 5:
            connectivity = np.load('connectivities/connectivity_google_sycamore_11.npy',allow_pickle=True).tolist()
        elif N == 6:
            connectivity = np.load('connectivities/connectivity_google_sycamore_13.npy',allow_pickle=True).tolist()
        else:
            connectivity = None
    else:
        connectivity = None

    coupling_map = connectivity

    if noise_model is None:
        density_matrix = False
    else:
        density_matrix = True

    os.makedirs(path, exist_ok=True)
    os.makedirs(path+'/training_states', exist_ok=True)

    backend = construct_backend("qibojit",platform=backend_name)
    backend.set_precision(precision)
    backend.set_threads(nthreads)

    if backend.platform == 'cupy':
        import cupy as cp

    model = XXZ_folded(N, M, D, domain_pos, backend)
    model._get_roots()
    circ_xx, circ_xxb = model.get_xx_b_circuit()
    circ_u0 = model.get_U0_circ()
    circ_d = model.get_D_circ()
    circ_Psi_M_0 = model.get_Psi_M_0_circ()

    circ = model.get_full_circ()
    #print(circ.nqubits)
    #print(circ.depth)
    # circ_quantinuum = model.circ_to_quantinuum(circ)
    # print('test')

    # from pytket.extensions.quantinuum import QuantinuumBackend
    # backend_q = QuantinuumBackend('H2-1SC')
    # circ_quantinuum = backend_q.get_compiled_circuit(circ_quantinuum, optimisation_level=3)
#############################3
    #print(circ_quantinuum.depth(), circ_quantinuum.n_1qb_gates(), circ_quantinuum.n_2qb_gates())
    circ_qiskit = model.circ_to_qiskit(circ)
    circ_qiskit1 = transpile(circ_qiskit,basis_gates=basis_gates,coupling_map=coupling_map,optimization_level=3,layout_method='trivial',routing_method='sabre')

    
    if circ_qiskit1.layout is None:
        layout_final = None
    else:
        layout_final = []
        for q in circ_qiskit1.layout.final_layout.get_virtual_bits().values():
            layout_final.append(q)
        print('final layout', layout_final)
    
    qasm_code = qasm2.dumps(circ_qiskit1)
    circ = Circuit.from_qasm(qasm_code)

    print('gate types', circ.gate_types)
    print('depht', circ.depth)
    print('nqubits', circ.nqubits)
    print('\n')

    model.circ_full = circ
###################################
    #layout_final = None 
    state_noiseless = model.get_state(density_matrix=False, boundaries=boundaries, layout=layout_final)

    # from pytket.extensions.qiskit import AerStateBackend
    # aer_state_b = AerStateBackend()
    # # circ_quantinuum = aer_state_b.get_compiled_circuit(circ_quantinuum)

    # state_handle = aer_state_b.process_circuit(circ_quantinuum)
    # statevector = aer_state_b.get_result(state_handle).get_state()

    # circ.density_matrix = False
    # print(fidelity(backend.execute_circuit(circ).state(), statevector, backend=backend))

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
    state_noise = model.get_state(density_matrix=density_matrix, boundaries=boundaries, noise_model=noise_model, layout=layout_final)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    if backend.platform == 'cupy':
        cp.get_default_memory_pool().free_all_blocks()

    fid = fidelity(state_noiseless, state_noise, backend=backend)

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

    if backend.platform == 'cupy':
        cp.get_default_memory_pool().free_all_blocks()

    circ = model.circ_full

    seed = None
    backend = None
    backend, local_state = _check_backend_and_local_state(seed, backend)

    model.backend = backend
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
    q1_0 = model.get_q1(boundaries)
    q2_0 = model.get_q2(boundaries)
    q1 = q1_0 - (model.N/2)*SymbolicHamiltonian(I(q1_0.nqubits-1), backend=backend)
    q2 = q2_0 - ((model.N+1)/2)*SymbolicHamiltonian(I(q2_0.nqubits-1), backend=backend)

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
            train_val["noiseless"],
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
