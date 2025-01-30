# Efficient Eigenstate Preparation in an Integrable Model with Hilbert Space Fragmentation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13925212.svg)](https://doi.org/10.5281/zenodo.13925211)

This repository contains the code to reproduce the numerical implementations presented in the manuscript [Efficient Eigenstate Preparation in an Integrable Model with Hilbert Space Fragmentation]().


## Dependences

- `Python>=3.12.8`

- `qibo==0.2.15`

- `qibojit==0.1.7`

- `qiskit==1.3.2`


## Usage
[`XXZ_folded.py`](https://github.com/AlejandroSopena/XXZ_folded/blob/main/XXZ_folded.py) contains a class to generate the `circuit` to prepare an eigenstate of the XXZ folded model with $N$ bulk sites $M$ magnons and $D$ domain walls.
For $N=5$ and $N=6$ with one magnon, the simplifications explained in the article are implemented.
```python
from XXZ_folded import XXZ_folded

N = 7
D = 2
M = 1
domain_pos = [[5,6]]

model = XXZ_folded(N, M, D, domain_pos)
model._get_roots()
circ_xx, circ_xxb = model.get_xx_b_circuit()
circ_u0 = model.get_U0_circ()
circ_d = model.get_D_circ()
circ_Psi_M_0 = model.get_Psi_M_0_circ()

circuit = model.get_full_circ()
```

[`run_simulation.py`](https://github.com/AlejandroSopena/XXZ_folded/blob/main/run_simulation.py) performs the simulations explained in the paper for a given eigenstate. The simulation can be customized throught specific command-line arguments.
```python
python run_simulation.py --path result --N 10 --M 1 --D 4 --domain_pos [[3,4],[7,8]] --connectivity google_sycamore --basis_gates cx rz sx x id --boundaries False --lamb 0.003 --n_training_samples 50 --precision single --backend numba
```
- `path`: path to save the data files with the expectaion values and the states.
- `N`: number of bulk qubits.
- `M`: number of magnons.
- `D` : number of domain walls.
- `domain_pos`: position of the domain walls.
- `connectivity`: can be `google_sycamore` for $N=5$ and $N=6$. `None` otherwise.
- `basis_gates`: single-qubit and two-qubit gates used to decompose the circuit.
- `boundaries`: if `True`, it extends the final state to $N+2$ qubits.
- `lamb`: depolarizing parameter used for the noisy simulation.
- `n_training_samples`: number of near Clifford circuits used in CDR.
- `backend`: Calculation engine: `numba` or `cupy`.
- `precision`: `single`: enables `complex64` and `double` enables `complex128`.
- `nthreads`: Number of threads for `numba`.

The structure of the output files is

- `path/state.npy`: dictionary containing the noiseless and noisy states under the keys `noisy` and `noiseless`, respectively.
- `path/training_states/training_circuits.npy`: list with the training set of near Clifford circuits used in CDR.
- `path/training_states/states_i.npy`: dictionary containing the noiseless and noisy i-th training state under the keys `noisy` and `noiseless`, respectively.
- `path/mitigated_values.npy`: list of lists. 
    The first element is `[[circuit,layout],energy_noiseless,Q1_noiseless,Q2_noiseless]`, where `circuit` is the circuit to prepare the eigenstate, `layout` denotes the mapping from virtual to physical qubits, and `energy_noiseless`, `Q1_noiseless`, `Q2_noiseless` are the noiseless values for the energy, $Q_1$, and $Q_2$, respectively.
    The second, third and fourth elements have the structure `[mit_val, val, optimal_params, train_val]`.
    The second element corresponds to the energy, the third to $Q_1$, and the fourth to $Q_2$.
    `mit_val` is the mitigated expectation value, `val`is the noisy expectation value, `optimal params` is a list with to elements `[a,b]` which are the optimal fit parameters from CDR and produce the line $ax+b$, 
    `tain_val`is a dictionary containing the noiseless and noisy training states under the keys `noisy` and `noiseless`, respectively.

The file `data.gz` contains the files to reproduce the results of the paper.
