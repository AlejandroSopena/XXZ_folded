import numpy as np
import pickle
import time

def _get_state(model, i, path, noise_model=None, boundaries=False, density_matrix=False, layout=None, backend=None):
    circuits = np.load(path+'/training_states/training_circuits.npy',allow_pickle=True)
    circuit = circuits[i]
    model.circ_full = circuit
    training_state = {}
    backend.set_precision('single')
    state1 = model.get_state(noise_model=None, boundaries=boundaries, density_matrix=False, layout=layout, backend=backend)
    state2 = model.get_state(noise_model=noise_model, boundaries=boundaries, density_matrix=density_matrix, layout=layout, backend=backend)

    training_state['noiseless'] = state1
    training_state['noisy'] = state2

    np.save(path+f'/training_states/states_{i}.npy', training_state)


if __name__ == "__main__":
    with open("input_args.pkl", "rb") as f:
        model, i, path, noise_model, boundaries, density_matrix, layout, backend = pickle.load(f)

    start_time = time.time()
    _get_state(model, i, path, noise_model, boundaries, density_matrix, layout, backend)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f" Elapsed time: {elapsed_time} seconds")
