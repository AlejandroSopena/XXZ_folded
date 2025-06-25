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
from qibo.noise import (
    NoiseModel,
    DepolarizingError,
    ThermalRelaxationError,
    ReadoutError,
)
from qibo.models.error_mitigation import sample_training_circuit_cdr
from qibo.backends import _check_backend_and_local_state, construct_backend
from qibo.symbols import I
from qibo.hamiltonians import SymbolicHamiltonian
from training_results_quantinuum import _get_state_cdr, get_mit_value_cdr
from utils_quantinuum import compile_quantinuum
from XXZ_folded import XXZ_folded
import qnexus as qnx
#qnx.login_with_credentials()
def main():

    def parse_nested_list(s):
        try:
            return eval(s)
        except:
            raise argparse.ArgumentTypeError("Invalid format for nested list")

    parser = argparse.ArgumentParser(
        description="Run simulation with specified parameters."
    )
    parser.add_argument(
        "--basis_gates",
        nargs="+",
        default=["cx", "rz", "sx", "x", "id"],
        help="List of basis gates",
    )
    parser.add_argument(
        "--boundaries", type=bool, default=False, help="Boundaries flag"
    )
    parser.add_argument("--error_detection", type=bool, help="Boundaries flag")
    parser.add_argument("--lamb", type=float, default=3e-3, help="Lambda value")
    parser.add_argument(
        "--n_training_samples", type=int, default=50, help="Number of training samples"
    )
    parser.add_argument(
        "--path", type=str, default="result_test_", help="Path to save states"
    )
    parser.add_argument("--N", type=int, default=7, help="Number of qubits")
    parser.add_argument("--M", type=int, default=1, help="Number of magnons")
    parser.add_argument("--D", type=int, default=2, help="Number of domain walls")
    parser.add_argument(
        "--momentum_ints", type=list, default=[], help="Momentum integers"
    )
    parser.add_argument(
        "--domain_pos",
        type=parse_nested_list,
        default=[[5, 6]],
        help="Domain positions",
    )
    parser.add_argument(
        "--connectivity", type=str, default=None, help="Connectivity type"
    )
    parser.add_argument(
        "--backend", type=str, default="numba", help="Calculation engine: numba or cupy"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="double",
        help="Precision type: double or single",
    )
    parser.add_argument(
        "--nthreads", type=int, default=8, help="Number of threads for numba"
    )
    parser.add_argument(
        "--device", type=str, default="local_noiseless_simulator", help="Quantinuum device (H1-1E, H2-1E). Default: local_noiseless_simulator"
    )
    parser.add_argument(
        "--nshots", type=int, default=1000, help="Number of shots for sampling"
    )

    args = parser.parse_args()

    basis_gates = args.basis_gates
    boundaries = args.boundaries
    error_detection = args.error_detection
    lamb = args.lamb
    n_training_samples = args.n_training_samples
    path = args.path
    N = args.N
    M = args.M
    D = args.D
    momentum_ints = args.momentum_ints
    domain_pos = args.domain_pos
    connectivity = args.connectivity
    backend_name = args.backend
    precision = args.precision
    nthreads = args.nthreads
    device = args.device
    nshots = args.nshots

    path = path + f"{device}_N{N}_M{M}_D{D}"
    momentum_ints = np.linspace(1, N - M - D + 1, M).tolist()
    #momentum_ints = [i + 1 for i in range(M)]

    if lamb == 0:
        noise_model = None
    else:
        noise_model = NoiseModel()
        noise_model.add(DepolarizingError(lamb), gates.CNOT)

        # p1_fault = 2.9e-5
        # p2_fault = 1.28e-3
        # p_meas_0 = 5e-4
        # p_meas_1 = 2.5e-3
        # p1_emision = 3.2e-1
        # p2_emision = 4.8e-1

        # t1 = 1/(p1_emision + p2_emision)
        # t2 = 2*t1
        # t_gate = 1e-3*t2
        # readout_matrix = np.array([[1-p_meas_0, p_meas_0], [p_meas_1, 1-p_meas_1]])
        # noise_model = NoiseModel()
        # noise_model.add(DepolarizingError(p2_fault),gates.RZZ)
        # noise_model.add(DepolarizingError(p1_fault),gates.U3)
        # noise_model.add(ThermalRelaxationError(t1, t2, t_gate), gates.RZZ)
        # noise_model.add(ReadoutError(readout_matrix), gates.M)

    if connectivity == "google_sycamore":
        if N == 5:
            connectivity = np.load(
                "connectivities/connectivity_google_sycamore_11.npy", allow_pickle=True
            ).tolist()
        elif N == 6:
            connectivity = np.load(
                "connectivities/connectivity_google_sycamore_13.npy", allow_pickle=True
            ).tolist()
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
    os.makedirs(path + "/training_states", exist_ok=True)
    os.makedirs(path + "/noise_states_zne", exist_ok=True)

    backend = construct_backend("qibojit", platform=backend_name)
    backend.set_precision(precision)
    backend.set_threads(nthreads)

    if backend.platform == "cupy":
        import cupy as cp

    model = XXZ_folded(N, M, D, momentum_ints, domain_pos, backend)

    model._get_roots()
    circ_xx, circ_xxb = model.get_xx_b_circuit()
    circ_u0 = model.get_U0_circ()
    circ_Psi_M_0 = model.get_Psi_M_0_circ()

    # if D == 0:
    #     circ = circ_Psi_M_0
    # else:
    #     circ_d = model.get_D_circ()
    #     circ = model.get_full_circ()

    if D != 0:
        circ_d = model.get_D_circ()

    circ = model.get_full_circ()
    # print(circ().symbolic())
    # print(circ.nqubits)
    # print(circ.depth)
    # circ_quantinuum = model.circ_to_quantinuum(circ)
    # print('test')

    # from pytket.extensions.quantinuum import QuantinuumBackend
    # backend_q = QuantinuumBackend('H2-1SC')
    # circ_quantinuum = backend_q.get_compiled_circuit(circ_quantinuum, optimisation_level=3)
    #############################3
    # print(circ_quantinuum.depth(), circ_quantinuum.n_1qb_gates(), circ_quantinuum.n_2qb_gates())

    circ_qiskit = model.circ_to_qiskit(circ)
    circ_qiskit1 = transpile(
        circ_qiskit,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        optimization_level=3,
        layout_method="trivial",
        routing_method="sabre",
    )

    if circ_qiskit1.layout is None:
        layout_final = None
    else:
        layout_final = []
        for q in circ_qiskit1.layout.final_layout.get_virtual_bits().values():
            layout_final.append(q)
        print("final layout", layout_final)

    qasm_code = qasm2.dumps(circ_qiskit1)
    circ_qibo = Circuit.from_qasm(qasm_code)

    print("gate types", circ.gate_types)
    print("depht", circ.depth)
    print("nqubits", circ.nqubits)
    print("\n")

    # from pytket.qasm import circuit_to_qasm_str
    # from pytket.extensions.quantinuum import QuantinuumBackend
    # device_backend = QuantinuumBackend('H2-1E')

    # circ_quantinuum = model.circ_to_quantinuum(circ, error_detection=error_detection)
    # circ_quantinuum = device_backend.get_compiled_circuit(circ_quantinuum, optimisation_level=2)
    # # qasm_code = circuit_to_qasm_str(circ_quantinuum)
    # # circ = Circuit.from_qasm(qasm_code)
    # from pytket.extensions.qiskit import tk_to_qiskit
    # circ_qiskit = tk_to_qiskit(circ_quantinuum)
    # qasm_code = qasm2.dumps(circ_qiskit)
    # circ = Circuit.from_qasm(qasm_code)

    model.circ_full = circ_qibo
    np.save(path + "/circuit.npy", circ_qibo)
    ###################################
    layout_final = None
    state_noiseless = model.get_state(
        density_matrix=False, boundaries=boundaries, layout=layout_final
    )

    # circ_quantinuum = model.circ_to_quantinuum(circ, error_detection=error_detection)
    # from pytket.extensions.qiskit import AerStateBackend
    # aer_state_b = AerStateBackend()
    # circ_quantinuum = aer_state_b.get_compiled_circuit(circ_quantinuum)

    # state_handle = aer_state_b.process_circuit(circ_quantinuum)
    # statevector = aer_state_b.get_result(state_handle).get_state()
    # circ.density_matrix = False
    # print(fidelity(backend.execute_circuit(circ).state(), statevector, backend=backend))

    energy_noiseless = model.get_energy(state_noiseless, boundaries=boundaries)
    Q1_noiseless = model.get_magnetization(state_noiseless, boundaries=boundaries)
    Q2_noiseless = model.get_correlation(state_noiseless, boundaries=boundaries)
    E1_noiseless = model.get_ej_expectation(1, state_noiseless, boundaries=boundaries)
    E2_noiseless = model.get_ej_expectation(2, state_noiseless, boundaries=boundaries)
    nonlocal_pauli_noiseless = model.get_nonlocal_pauli_expectation(
        3, state_noiseless, boundaries=boundaries
    )

    if backend.platform == "cupy":
        energy_noiseless = float(energy_noiseless.get())
        Q1_noiseless = float(Q1_noiseless.get())
        Q2_noiseless = float(Q2_noiseless.get())

    fid_ham, fid_q1, fid_q2 = model.check_fidelity(
        state_noiseless, boundaries=boundaries
    )
    print("  Fidelity Hamiltonian: ", fid_ham)
    print("  Fidelity Q1: ", fid_q1)
    print("  Fidelity Q2: ", fid_q2)

    print("Noiseless")
    print("  Energy: ", energy_noiseless)
    print("  Q1: ", Q1_noiseless)
    print("  Q2: ", Q2_noiseless)
    print("  E1: ", E1_noiseless)
    print("  E2: ", E2_noiseless)
    print("  Nonlocal Pauli: ", nonlocal_pauli_noiseless)

    # counts_x, counts_y, counts_z, counts_energy = model.sample_circuit(nshots, noise_model, layout_final, boundaries=boundaries, error_detection=error_detection, backend=backend)

    # shots_x = np.sum(list(counts_x.values()))
    # shots_y = np.sum(list(counts_y.values()))
    # shots_z = np.sum(list(counts_z.values()))
    # print('shots_x', shots_x, 'shots_y', shots_y, 'shots_z', shots_z)

    # print(counts_z)
    # from pytket.extensions.quantinuum import QuantinuumBackend
    # device_backend = QuantinuumBackend('H1-1E')
    # from pytket.extensions.qiskit import AerBackend
    # device_backend = AerBackend()

    measure_all = True
    error_detection = True
    mitigation_method = "CDR_ZNE"

    # if "ZNE" in mitigation_method:
    #     circ_quantinuum = model.circ_to_quantinuum(circ_qibo,measure_all=measure_all)    
    #     optimization_level = 3
    #     name_project = "XXZ_folded"    
    #     circ_compiled = compile_quantinuum([circ_quantinuum], name_project, optimization_level, nshots, device, compile=True, counts=False, execute=False)[0]
    #     compile = False
    #     circ_to_quantinuum = False
    #     model.circ_full = circ_compiled
    #     np.save(path + "/circuit_compiled.npy", circ_compiled)
    compile = True
    circ_to_quantinuum = True
    counts_x1, counts_y1, counts_z1, counts_energy1, compiled_circuits = model.sample_circuit_quantinuum(
        device, nshots, layout_final, boundaries=boundaries, measure_all=measure_all, compile=compile, circ_to_quantinuum=circ_to_quantinuum
    )
    new_indices_list, counts_zxxz_list1, counts_zyyz_list1 = counts_energy1
    if measure_all:
        counts_x = model.get_nsites_counts(counts_x1, postselect=False)
        counts_y = model.get_nsites_counts(counts_y1, postselect=False)
        counts_z = model.get_nsites_counts(counts_z1, postselect=False)
        counts_zxxz_list = [
            model.get_nsites_counts(counts, postselect=False)
            for counts in counts_zxxz_list1
        ]
        counts_zyyz_list = [
            model.get_nsites_counts(counts, postselect=False)
            for counts in counts_zyyz_list1
        ]
    else:
        counts_x = counts_x1
        counts_y = counts_y1
        counts_z = counts_z1
        counts_zxxz_list = counts_zxxz_list1
        counts_zyyz_list = counts_zyyz_list1

    # print(counts_z)

    q1_sample = model.sample_q1(counts_z, boundaries=boundaries)
    q2_sample = model.sample_q2(counts_z, boundaries=boundaries)
    # energy_sample = model.sample_energy(counts_x, counts_y, nshots, noise_model, layout=layout_final, boundaries=boundaries, backend=backend)

    e1_sample = model.sample_ej(1, counts_z, boundaries=boundaries)
    e2_sample = model.sample_ej(2, counts_z, boundaries=boundaries)
    nonlocal_pauli_sample = model.sample_nonlocal_pauli(
        3, counts_z, boundaries=boundaries
    )
    energy_sample = model.sample_energy(
        counts_x,
        counts_y,
        new_indices_list,
        counts_zxxz_list,
        counts_zyyz_list,
        backend=backend,
    )

    print("  Energy sample: ", energy_sample)
    print("  Q1 sample: ", q1_sample)
    print("  Q2 sample: ", q2_sample)
    print("  E1 sample: ", e1_sample)
    print("  E2 sample: ", e2_sample)
    print("  Nonlocal Pauli sample: ", nonlocal_pauli_sample)

    if measure_all and error_detection:
        counts_x_post = model.get_nsites_counts(counts_x1, postselect=True)
        counts_y_post = model.get_nsites_counts(counts_y1, postselect=True)
        counts_z_post = model.get_nsites_counts(counts_z1, postselect=True)
        counts_zxxz_post_list = [
            model.get_nsites_counts(counts, postselect=True)
            for counts in counts_zxxz_list1
        ]
        counts_zyyz_post_list = [
            model.get_nsites_counts(counts, postselect=True)
            for counts in counts_zyyz_list1
        ]

        q1_sample_post = model.sample_q1(counts_z_post, boundaries=boundaries)
        q2_sample_post = model.sample_q2(counts_z_post, boundaries=boundaries)
        e1_sample_post = model.sample_ej(1, counts_z_post, boundaries=boundaries)
        e2_sample_post = model.sample_ej(2, counts_z_post, boundaries=boundaries)
        nonlocal_pauli_sample_post = model.sample_nonlocal_pauli(
            3, counts_z_post, boundaries=boundaries
        )
        energy_sample_post = model.sample_energy(
            counts_x_post,
            counts_y_post,
            new_indices_list,
            counts_zxxz_post_list,
            counts_zyyz_post_list,
            backend=backend,
        )

        print("Error detection")
        print("  Energy sample post: ", energy_sample_post)
        print("  Q1 sample post: ", q1_sample_post)
        print("  Q2 sample post: ", q2_sample_post)
        print("  E1 sample post: ", e1_sample_post)
        print("  E2 sample post: ", e2_sample_post)
        print("  Nonlocal Pauli sample post: ", nonlocal_pauli_sample_post)

    if error_detection:
        counts_result = [[counts_x, counts_x_post, counts_x1], [counts_y, counts_y_post, counts_y1], [counts_z, counts_z_post, counts_z1], [counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1], [counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1], new_indices_list]
    else:
        counts_result = [counts_x, counts_y, counts_z, counts_zxxz_list, counts_zyyz_list, new_indices_list]
    np.save(path + "/state.npy", {"noiseless": state_noiseless, "noisy": [counts_result, compiled_circuits]})

    if backend.platform == "cupy":
        cp.get_default_memory_pool().free_all_blocks()

    #circ = model.circ_full
    


    seed = None
    backend = None
    backend, local_state = _check_backend_and_local_state(seed, backend)

    model.backend = backend
    
    if "CDR" in mitigation_method:
        # CDR         
        training_circuits = [
            sample_training_circuit_cdr(circ_qibo, seed=local_state, backend=backend)
            for _ in range(n_training_samples)
        ]

        np.save(path + "/training_states/training_circuits.npy", training_circuits)

        for i in range(n_training_samples):
            print("Training circuit: ", i)
            _get_state_cdr(model, i, path, device, nshots, measure_all, error_detection, boundaries, layout_final, backend, compile=True)


    
    states = np.load(path + "/state.npy", allow_pickle=True).item()
    noisy_state = states["noisy"]   
   



    observables_label = ["Energy", "Q1", "Q2", "E1", "E2", "Nonlocal Pauli"]
    noiseless_val_list = [energy_noiseless, Q1_noiseless, Q2_noiseless, E1_noiseless, E2_noiseless, nonlocal_pauli_noiseless]

    results = [[[circ, layout_final], energy_noiseless, Q1_noiseless, Q2_noiseless, E1_noiseless, E2_noiseless, nonlocal_pauli_noiseless]]
    if "CDR" in mitigation_method:
        results_cdr = results.copy()

    for i, observable_label in enumerate(observables_label):
        if "CDR" in mitigation_method:
            print("CDR")
            # mit_val, mit_val_post, val, val_post, optimal_params, optimal_params_post, train_val = get_mit_value_cdr(
            #     observable_label, n_training_samples, noisy_state
            # )
            mit_val, mit_val_post, val, val_post, optimal_params, optimal_params_post, train_val = get_mit_value_cdr(
                model, observable_label, n_training_samples, noisy_state, boundaries, path, backend, local_state
            )
            results_cdr.append([mit_val, mit_val_post, val, val_post, optimal_params, optimal_params_post, train_val, noiseless_val_list[i]])
            print(observable_label)
            print("  Mitigated: ", mit_val)
            print("  Mitigated post: ", mit_val_post)
            print("  Noisy: ", val)
            print("  Noisy post: ", val_post)
            print("  Noiseless: ", noiseless_val_list[i])
            print("  Optimal parameters: ", optimal_params)
            print("  Optimal parameters post: ", optimal_params_post)
            print("  Noise circuit values: ", train_val) 

        
    if "CDR" in mitigation_method:
        results_cdr = np.array(results_cdr, object)
        np.save(path + f"/mitigated_values_CDR.npy", results_cdr, allow_pickle=True)


if __name__ == "__main__":
    main()
