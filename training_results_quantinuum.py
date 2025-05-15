import numpy as np
import pickle
import time
from scipy.optimize import curve_fit

def _get_state_cdr(model, i, path, device, nshots, measure_all, error_detection, boundaries=False, layout=None, backend=None, compile=True):
    circuits = np.load(path+'/training_states/training_circuits.npy',allow_pickle=True)
    circuit = circuits[i]
    model.circ_full = circuit
    training_state = {}
    backend.set_precision('single')
    state1 = model.get_state(noise_model=None, boundaries=boundaries, density_matrix=False, layout=layout)
    #state2 = model.get_state(noise_model=noise_model, boundaries=boundaries, density_matrix=density_matrix, layout=layout)

    counts_x1, counts_y1, counts_z1, counts_energy1, compiled_circuits = model.sample_circuit_quantinuum(
        device, nshots, layout, boundaries=boundaries, measure_all=measure_all, compile=compile
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

    training_state['noiseless'] = state1
    if error_detection:
        counts_result = [[counts_x, counts_x_post, counts_x1], [counts_y, counts_y_post, counts_y1], [counts_z, counts_z_post, counts_z1], [counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1], [counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1], new_indices_list]
    else:
        counts_result = [counts_x, counts_y, counts_z, counts_zxxz_list, counts_zyyz_list, new_indices_list]
    training_state['noisy'] = [counts_result, compiled_circuits] 
    np.save(path+f'/training_states/states_{i}.npy', training_state)

def _get_state_cdr_reduced_counts(model, n_training_samples, nshots, path, measure_all=True, error_detection=True, return_data=False):   
    counts_results = []
    training_states = {'noisy': []}
    for i in range(n_training_samples):
        training_state = np.load(path+f'/training_states/states_{i}.npy',allow_pickle=True).item()
        counts = training_state['noisy'][0]
        counts_x, counts_x_post, counts_x1 = counts[0]
        counts_y, counts_y_post, counts_y1 = counts[1]
        counts_z, counts_z_post, counts_z1 = counts[2]
        counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1 = counts[3]
        counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1 = counts[4]
        new_indices_list = counts[5]
        total_shots = sum(counts_x.values())
        if nshots < total_shots:
            counts_x1 = get_reduced_counts(counts_x1, nshots)
            counts_y1 = get_reduced_counts(counts_y1, nshots)
            counts_z1 = get_reduced_counts(counts_z1, nshots)
            counts_zxxz_list1 = [get_reduced_counts(counts, nshots) for counts in counts_zxxz_list1]
            counts_zyyz_list1 = [get_reduced_counts(counts, nshots) for counts in counts_zyyz_list1]

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

        if error_detection:
            counts_result = [[counts_x, counts_x_post, counts_x1], [counts_y, counts_y_post, counts_y1], [counts_z, counts_z_post, counts_z1], [counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1], [counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1], new_indices_list]
        else:
            counts_result = [counts_x, counts_y, counts_z, counts_zxxz_list, counts_zyyz_list, new_indices_list]
        counts_results.append(counts_result)

    training_states['noisy'] = counts_results
    np.save(path+f'/training_states/reduced_shots_{nshots}.npy', training_states)
    if return_data:
        return training_states


def _get_state_zne(model, i, path, device, nshots, measure_all, error_detection, boundaries=False, layout=None, backend=None, compile=False):
    circuits = np.load(path+"/noise_states_zne/noise_circuits.npy", allow_pickle=True)
    circuit = circuits[i]
    model.circ_full = circuit
    noise_state = {}
    backend.set_precision('single')

    counts_x1, counts_y1, counts_z1, counts_energy1, compiled_circuits = model.sample_circuit_quantinuum(
        device, nshots, layout, boundaries=boundaries, measure_all=measure_all, compile=False, circ_to_quantinuum=False, optimization_level=0
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

    if error_detection:
        counts_result = [[counts_x, counts_x_post, counts_x1], [counts_y, counts_y_post, counts_y1], [counts_z, counts_z_post, counts_z1], [counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1], [counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1], new_indices_list]
    else:
        counts_result = [counts_x, counts_y, counts_z, counts_zxxz_list, counts_zyyz_list, new_indices_list]
    noise_state['noisy'] = [counts_result, compiled_circuits] 
    np.save(path+f'/noise_states_zne/states_{i}.npy', noise_state)

def _get_state_zne_reduced_counts(model, noise_levels, nshots, path, measure_all=True, error_detection=True, return_data=False):   
    counts_results = []
    noise_states = {'noisy': []}
    for i in range(len(noise_levels)):
        noise_state = np.load(path+f'/noise_states_zne/states_{i}.npy',allow_pickle=True).item()
        counts = noise_state['noisy'][0]
        counts_x, counts_x_post, counts_x1 = counts[0]
        counts_y, counts_y_post, counts_y1 = counts[1]
        counts_z, counts_z_post, counts_z1 = counts[2]
        counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1 = counts[3]
        counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1 = counts[4]
        new_indices_list = counts[5]
        total_shots = sum(counts_x.values())
        if nshots < total_shots:
            counts_x1 = get_reduced_counts(counts_x1, nshots)
            counts_y1 = get_reduced_counts(counts_y1, nshots)
            counts_z1 = get_reduced_counts(counts_z1, nshots)
            counts_zxxz_list1 = [get_reduced_counts(counts, nshots) for counts in counts_zxxz_list1]
            counts_zyyz_list1 = [get_reduced_counts(counts, nshots) for counts in counts_zyyz_list1]

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

        if error_detection:
            counts_result = [[counts_x, counts_x_post, counts_x1], [counts_y, counts_y_post, counts_y1], [counts_z, counts_z_post, counts_z1], [counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1], [counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1], new_indices_list]
        else:
            counts_result = [counts_x, counts_y, counts_z, counts_zxxz_list, counts_zyyz_list, new_indices_list]
        counts_results.append(counts_result)

    noise_state['noisy'] = counts_results
    np.save(path+f'/noise_states_zne/reduced_shots_{nshots}.npy', noise_state)
    if return_data:
        return noise_state

def _get_state_reduced_counts(model, nshots, path, measure_all=True, error_detection=True, return_data=False):   
    new_state = {}
    state = np.load(path+f'/state.npy',allow_pickle=True).item()
    counts = state['noisy'][0]
    counts_x, counts_x_post, counts_x1 = counts[0]
    counts_y, counts_y_post, counts_y1 = counts[1]
    counts_z, counts_z_post, counts_z1 = counts[2]
    counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1 = counts[3]
    counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1 = counts[4]
    new_indices_list = counts[5]
    total_shots = sum(counts_x.values())
    if nshots < total_shots:
        counts_x1 = get_reduced_counts(counts_x1, nshots)
        counts_y1 = get_reduced_counts(counts_y1, nshots)
        counts_z1 = get_reduced_counts(counts_z1, nshots)
        counts_zxxz_list1 = [get_reduced_counts(counts, nshots) for counts in counts_zxxz_list1]
        counts_zyyz_list1 = [get_reduced_counts(counts, nshots) for counts in counts_zyyz_list1]

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

    if error_detection:
        counts_result = [[counts_x, counts_x_post, counts_x1], [counts_y, counts_y_post, counts_y1], [counts_z, counts_z_post, counts_z1], [counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1], [counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1], new_indices_list]
    else:
        counts_result = [counts_x, counts_y, counts_z, counts_zxxz_list, counts_zyyz_list, new_indices_list]


    new_state['noisy'] = [counts_result]
    np.save(path+f'/reduced_shots_{nshots}.npy', new_state)
    if return_data:
        return new_state

def get_reduced_counts(counts, nshots):
    import random
    from collections import Counter

    total_nshots = sum(counts.values())
    probs = np.array(list(counts.values())) / total_nshots
    keys = np.array(list(counts.keys()))
    samples = random.choices(keys, probs, k=nshots)
    new_counts = Counter(samples)

    return new_counts

def get_mit_value_cdr(model, observable_label, n_training_samples, noisy_state, boundaries, path, backend, local_state, nshots=None, input_data=None):
#pass noisy state with nshots
    if observable_label == "Energy":
        observable = model.get_xxz_folded_hamiltonian(boundaries)
    elif observable_label == "Q1":
        observable = model.get_q1(boundaries)
    elif observable_label == "Q2":
        observable = model.get_q2(boundaries)
    elif observable_label == "E1":
        observable = model.get_ej(1, boundaries)
    elif observable_label == "E2":
        observable = model.get_ej(2, boundaries)
    elif observable_label == "Nonlocal Pauli":
        observable = model.get_nonlocal_pauli(3, boundaries)

    train_val = {"noiseless": [], "noisy": []}
    for i in range(n_training_samples):
        training_state = np.load(
            path + f"/training_states/states_{i}.npy", allow_pickle=True
        ).item()
        state = training_state["noiseless"]

        val = observable.expectation(state)
        if observable_label == "Q1":
            val = val - (model.N / 2)
        elif observable_label == "Q2":
            val = val - ((model.N + 1) / 2)
        elif observable_label == "E1":
            val = val - (1 / 2**1)
        elif observable_label == "E2":
            val = val - (1 / 2**2)

        if backend.platform == "cupy":
            val = float(val.get())
        train_val["noiseless"].append(val)

        if nshots is not None:
            if input_data is None:
                training_state = np.load(
                    path + f"/training_states/reduced_shots_{nshots}.npy", allow_pickle=True
                ).item()['noisy'][i]
            else:
                training_state = input_data['noisy'][i]

            training_state = {"noisy": [training_state]}

        counts = training_state["noisy"][0]
        counts_x, counts_x_post, counts_x1 = counts[0]
        counts_y, counts_y_post, counts_y1 = counts[1]
        counts_z, counts_z_post, counts_z1 = counts[2]
        counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_post_list1 = counts[3]
        counts_zyyz_list, counts_zyyz_post_list, counts_zxxz_post_list1 = counts[4]
        new_indices_list = counts[5]
        train_nshots = sum(counts_x.values())

        if observable_label == "Energy":
            val = model.sample_energy(
                counts_x,
                counts_y,
                new_indices_list,
                counts_zxxz_list,
                counts_zyyz_list,
                backend=backend,
                )
            val_post = model.sample_energy(
                counts_x_post,
                counts_y_post,
                new_indices_list,
                counts_zxxz_post_list,
                counts_zyyz_post_list,
                backend=backend,
                )
        elif observable_label == "Q1":
            val = model.sample_q1(counts_z, boundaries=boundaries) - (model.N / 2)
            val_post = model.sample_q1(counts_z_post, boundaries=boundaries) - (model.N / 2)
        elif observable_label == "Q2":
            val = model.sample_q2(counts_z, boundaries=boundaries) - ((model.N + 1) / 2)
            val_post = model.sample_q2(counts_z_post, boundaries=boundaries) - ((model.N + 1) / 2)
        elif observable_label == "E1":
            val = model.sample_ej(1, counts_z, boundaries=boundaries) - (1 / 2**1)
            val_post = model.sample_ej(1, counts_z_post, boundaries=boundaries) - (1 / 2**1)
        elif observable_label == "E2":
            val = model.sample_ej(2, counts_z, boundaries=boundaries) - (1 / 2**2)
            val_post = model.sample_ej(2, counts_z_post, boundaries=boundaries) - (1 / 2**2)
        elif observable_label == "Nonlocal Pauli":
            val = model.sample_nonlocal_pauli(3, counts_z, boundaries=boundaries)
            val_post = model.sample_nonlocal_pauli(3, counts_z_post, boundaries=boundaries)

        if backend.platform == "cupy":
            val = float(val.get())
            val_post = float(val_post.get())
        train_val["noisy"].append([val, val_post])

    nparams = 2

    params = local_state.random(nparams)
    f = lambda x, a, b: a * x + b
    train_val["noisy"] = np.array(train_val["noisy"], object)
    optimal_params = curve_fit(
        f,
        train_val["noisy"][:, 0],
        train_val["noiseless"],
        p0=params,
    )[0]

    optimal_params_post = curve_fit(
        f,
        train_val["noisy"][:, 1],
        train_val["noiseless"],
        p0=params,
    )[0]

    counts = noisy_state[0]
    counts_x, counts_x_post, counts_x1 = counts[0]
    counts_y, counts_y_post, counts_y1 = counts[1]
    counts_z, counts_z_post, counts_z1 = counts[2]
    counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1 = counts[3]
    counts_zyyz_list, counts_zyyz_post_list, counts_zxxz_list1 = counts[4]
    new_indices_list = counts[5]
    if sum(counts_x.values()) != train_nshots:
        raise ValueError("Number of counts of the noisy state must be the same as nshots")

    if observable_label == "Energy":
        val = model.sample_energy(
            counts_x,
            counts_y,
            new_indices_list,
            counts_zxxz_list,
            counts_zyyz_list,
            backend=backend,
            )
        val_post = model.sample_energy(
            counts_x_post,
            counts_y_post,
            new_indices_list,
            counts_zxxz_post_list,
            counts_zyyz_post_list,
            backend=backend,
            )
    elif observable_label == "Q1":
        val = model.sample_q1(counts_z, boundaries=boundaries) - (model.N / 2)
        val_post = model.sample_q1(counts_z_post, boundaries=boundaries) - (model.N / 2)
    elif observable_label == "Q2":
        val = model.sample_q2(counts_z, boundaries=boundaries) - ((model.N + 1) / 2)
        val_post = model.sample_q2(counts_z_post, boundaries=boundaries) - ((model.N + 1) / 2)
    elif observable_label == "E1":
        val = model.sample_ej(1, counts_z, boundaries=boundaries) - (1 / 2**1)
        val_post = model.sample_ej(1, counts_z_post, boundaries=boundaries) - (1 / 2**1)
    elif observable_label == "E2":
        val = model.sample_ej(2, counts_z, boundaries=boundaries) - (1 / 2**2)
        val_post = model.sample_ej(2, counts_z_post, boundaries=boundaries) - (1 / 2**2)
    elif observable_label == "Nonlocal Pauli":
        val = model.sample_nonlocal_pauli(3, counts_z, boundaries=boundaries)
        val_post = model.sample_nonlocal_pauli(3, counts_z_post, boundaries=boundaries)


    if backend.platform == "cupy":
        val = float(val.get())
        val_post = float(val_post.get())
    mit_val = f(val, *optimal_params)
    mit_val_post = f(val_post, *optimal_params_post)

    if observable_label == "Q1":
        val = val + (model.N / 2)
        val_post = val_post + (model.N / 2)
        mit_val = mit_val + (model.N / 2)
        mit_val_post = mit_val_post + (model.N / 2)
    elif observable_label == "Q2":
        val = val + ((model.N + 1) / 2)
        val_post = val_post + ((model.N + 1) / 2)
        mit_val = mit_val + ((model.N + 1) / 2)
        mit_val_post = mit_val_post + ((model.N + 1) / 2)
    elif observable_label == "E1":
        val = val + (1 / 2**1)
        val_post = val_post + (1 / 2**1)
        mit_val = mit_val + (1 / 2**1)
        mit_val_post = mit_val_post + (1 / 2**1)
    elif observable_label == "E2":
        val = val + (1 / 2**2)
        val_post = val_post + (1 / 2**2)
        mit_val = mit_val + (1 / 2**2)
        mit_val_post = mit_val_post + (1 / 2**2)
    # train_val contains the expectation values of the noiseless and noisy states without constants to learn directly the model.
    # mit_val, mit_val_post, val, val_post are the mitigated and noisy values with the constants added.
    return mit_val, mit_val_post, val, val_post, optimal_params, optimal_params_post, train_val


def get_mit_value_zne(model, observable_label, noise_levels, noisy_state, boundaries, path, backend, local_state, nshots=None, input_data=None):

    if observable_label == "Energy":
        observable = model.get_xxz_folded_hamiltonian(boundaries)
    elif observable_label == "Q1":
        observable = model.get_q1(boundaries)
    elif observable_label == "Q2":
        observable = model.get_q2(boundaries)
    elif observable_label == "E1":
        observable = model.get_ej(1, boundaries)
    elif observable_label == "E2":
        observable = model.get_ej(2, boundaries)
    elif observable_label == "Nonlocal Pauli":
        observable = model.get_nonlocal_pauli(3, boundaries)

    train_val = {"noisy": []}
    for i in range(len(noise_levels)):

        if nshots is None:
            training_state = np.load(
                path + f"/noise_states_zne/states_{i}.npy", allow_pickle=True
                ).item()
        else:
            if input_data is None:
                training_state = np.load(
                    path + f"/noise_states_zne/reduced_shots_{nshots}.npy", allow_pickle=True
                ).item()["noisy"][i]
            else:
                training_state = input_data["noisy"][i]
            training_state = {"noisy": [training_state]}

        
        counts = training_state["noisy"][0]
        counts_x, counts_x_post, counts_x1 = counts[0]
        counts_y, counts_y_post, counts_y1 = counts[1]
        counts_z, counts_z_post, counts_z1 = counts[2]
        counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1 = counts[3]
        counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1 = counts[4]
        new_indices_list = counts[5]
        noise_levels_nshots = sum(counts_x.values())

        if observable_label == "Energy":
            val = model.sample_energy(
                counts_x,
                counts_y,
                new_indices_list,
                counts_zxxz_list,
                counts_zyyz_list,
                backend=backend,
                )
            val_post = model.sample_energy(
                counts_x_post,
                counts_y_post,
                new_indices_list,
                counts_zxxz_post_list,
                counts_zyyz_post_list,
                backend=backend,
                )
        elif observable_label == "Q1":
            val = model.sample_q1(counts_z, boundaries=boundaries) - (model.N / 2)
            val_post = model.sample_q1(counts_z_post, boundaries=boundaries) - (model.N / 2)
        elif observable_label == "Q2":
            val = model.sample_q2(counts_z, boundaries=boundaries) - ((model.N + 1) / 2)
            val_post = model.sample_q2(counts_z_post, boundaries=boundaries) - ((model.N + 1) / 2)
        elif observable_label == "E1":
            val = model.sample_ej(1, counts_z, boundaries=boundaries) - (1 / 2**1)
            val_post = model.sample_ej(1, counts_z_post, boundaries=boundaries) - (1 / 2**1)
        elif observable_label == "E2":
            val = model.sample_ej(2, counts_z, boundaries=boundaries) - (1 / 2**2)
            val_post = model.sample_ej(2, counts_z_post, boundaries=boundaries) - (1 / 2**2)
        elif observable_label == "Nonlocal Pauli":
            val = model.sample_nonlocal_pauli(3, counts_z, boundaries=boundaries)
            val_post = model.sample_nonlocal_pauli(3, counts_z_post, boundaries=boundaries)

        if backend.platform == "cupy":
            val = float(val.get())
            val_post = float(val_post.get())
        train_val["noisy"].append([val, val_post])


    counts = noisy_state[0]
    counts_x, counts_x_post, counts_x1 = counts[0]
    counts_y, counts_y_post, counts_y1 = counts[1]
    counts_z, counts_z_post, counts_z1 = counts[2]
    counts_zxxz_list, counts_zxxz_post_list, counts_zxxz_list1 = counts[3]
    counts_zyyz_list, counts_zyyz_post_list, counts_zyyz_list1 = counts[4]
    new_indices_list = counts[5]
    if sum(counts_x.values()) != noise_levels_nshots:
        raise ValueError("Number of counts of the noisy state must be the same as nshots")

    if observable_label == "Energy":
        val = model.sample_energy(
            counts_x,
            counts_y,
            new_indices_list,
            counts_zxxz_list,
            counts_zyyz_list,
            backend=backend,
            )
        val_post = model.sample_energy(
            counts_x_post,
            counts_y_post,
            new_indices_list,
            counts_zxxz_post_list,
            counts_zyyz_post_list,
            backend=backend,
            )
    elif observable_label == "Q1":
        val = model.sample_q1(counts_z, boundaries=boundaries) - (model.N / 2)
        val_post = model.sample_q1(counts_z_post, boundaries=boundaries) - (model.N / 2)
    elif observable_label == "Q2":
        val = model.sample_q2(counts_z, boundaries=boundaries) - ((model.N + 1) / 2)
        val_post = model.sample_q2(counts_z_post, boundaries=boundaries) - ((model.N + 1) / 2)
    elif observable_label == "E1":
        val = model.sample_ej(1, counts_z, boundaries=boundaries) - (1 / 2**1)
        val_post = model.sample_ej(1, counts_z_post, boundaries=boundaries) - (1 / 2**1)
    elif observable_label == "E2":
        val = model.sample_ej(2, counts_z, boundaries=boundaries) - (1 / 2**2)
        val_post = model.sample_ej(2, counts_z_post, boundaries=boundaries) - (1 / 2**2)
    elif observable_label == "Nonlocal Pauli":
        val = model.sample_nonlocal_pauli(3, counts_z, boundaries=boundaries)
        val_post = model.sample_nonlocal_pauli(3, counts_z_post, boundaries=boundaries)


    if backend.platform == "cupy":
        val = float(val.get())
        val_post = float(val_post.get())


    nparams = 2
    eps = 9e-4

    params = local_state.random(nparams)
    f = lambda x, a, b: a * x + b
    train_val["noisy"] = np.array(train_val["noisy"], object)
    optimal_params = curve_fit(
        f,
        np.array([1]+noise_levels)*eps,
        [val]+list(train_val["noisy"][:, 0]),
        p0=params,
    )[0]

    optimal_params_post = curve_fit(
        f,
        np.array([1]+noise_levels)*eps,
        [val_post]+list(train_val["noisy"][:, 1]),
        p0=params,
    )[0]

    mit_val = f(0, *optimal_params)
    mit_val_post = f(0, *optimal_params_post)

    if observable_label == "Q1":
        val = val + (model.N / 2)
        val_post = val_post + (model.N / 2)
        mit_val = mit_val + (model.N / 2)
        mit_val_post = mit_val_post + (model.N / 2)
    elif observable_label == "Q2":
        val = val + ((model.N + 1) / 2)
        val_post = val_post + ((model.N + 1) / 2)
        mit_val = mit_val + ((model.N + 1) / 2)
        mit_val_post = mit_val_post + ((model.N + 1) / 2)
    elif observable_label == "E1":
        val = val + (1 / 2**1)
        val_post = val_post + (1 / 2**1)
        mit_val = mit_val + (1 / 2**1)
        mit_val_post = mit_val_post + (1 / 2**1)
    elif observable_label == "E2":
        val = val + (1 / 2**2)
        val_post = val_post + (1 / 2**2)
        mit_val = mit_val + (1 / 2**2)
        mit_val_post = mit_val_post + (1 / 2**2)
    # train_val contains the expectation values of the noiseless and noisy states without constants to learn directly the model.
    # mit_val, mit_val_post, val, val_post are the mitigated and noisy values with the constants added.
    return mit_val, mit_val_post, val, val_post, optimal_params, optimal_params_post, train_val


# if __name__ == "__main__":
#     with open("input_args.pkl", "rb") as f:
#         model, i, path, device, nshots, measure_all, error_detection, boundaries, layout, backend = pickle.load(f)

#     start_time = time.time()
#     _get_state(model, i, path, device, nshots, measure_all, error_detection, boundaries, layout, backend)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f" Elapsed time: {elapsed_time} seconds")
