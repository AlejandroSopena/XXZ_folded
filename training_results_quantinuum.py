import numpy as np
import pickle
import time

def _get_state(model, i, path, device, nshots, measure_all, error_detection, boundaries=False, layout=None, backend=None):
    circuits = np.load(path+'/training_states/training_circuits.npy',allow_pickle=True)
    circuit = circuits[i]
    model.circ_full = circuit
    training_state = {}
    backend.set_precision('single')
    state1 = model.get_state(noise_model=None, boundaries=boundaries, density_matrix=False, layout=layout)
    #state2 = model.get_state(noise_model=noise_model, boundaries=boundaries, density_matrix=density_matrix, layout=layout)

    counts_x1, counts_y1, counts_z1, counts_energy1 = model.sample_circuit_quantinuum(
        device, nshots, layout, boundaries=boundaries, measure_all=measure_all
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
        training_state['noisy'] = [[counts_x, counts_x_post], [counts_y, counts_y_post], [counts_z, counts_z_post], [counts_zxxz_list, counts_zxxz_post_list], [counts_zyyz_list, counts_zyyz_post_list], new_indices_list]
    else:
        training_state['noisy'] = [counts_x, counts_y, counts_z, counts_zxxz_list, counts_zyyz_list, new_indices_list]

    np.save(path+f'/training_states/states_{i}.npy', training_state)


# if __name__ == "__main__":
#     with open("input_args.pkl", "rb") as f:
#         model, i, path, device, nshots, measure_all, error_detection, boundaries, layout, backend = pickle.load(f)

#     start_time = time.time()
#     _get_state(model, i, path, device, nshots, measure_all, error_detection, boundaries, layout, backend)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f" Elapsed time: {elapsed_time} seconds")
