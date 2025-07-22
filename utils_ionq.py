from qiskit import transpile
from qiskit_ionq import IonQProvider
import os


# def counts_to_qibo(counts):
#     new_counts = Counter()
#     for key, value in counts.items():
#         new_key = ''
#         for i in range(len(key)):
#             new_key += str(key[i])
#         new_counts[new_key] = value
#     return new_counts

def get_num_gates(circuit):
    single_qubit = 0
    two_qubit = 0
    for instruction in circuit.data:
        gate = instruction.operation
        if gate.name != 'measure' and gate.name != 'barrier':
            num_qubits = gate.num_qubits 
            if num_qubits == 1:
                single_qubit += 1
            elif num_qubits == 2:
                two_qubit += 1
    return [single_qubit, two_qubit]

def compile_ionq(circuits, optimization_level, nshots, device, compile=True, counts=False, execute=True):

    my_api_key = os.getenv("MY_IONQ_API_KEY")
    provider = IonQProvider(my_api_key)

    if compile:
        gateset = "native" 
    else:
        gateset = "QIS"

    if 'simulator' in device:
        backend = provider.get_backend("simulator", gateset=gateset)
        if 'noiseless' not in device:
            backend.set_options(noise_model=device[15::]) # ionq.simulator.aria-1
    else:
        backend = provider.get_backend(device[5::], gateset=gateset) # ionq.qpu.aria-1

    for j in range(len(circuits)):
        circuits[j].name = f"circuit_{j}" 
    #if compile:
    compiled_circuits = transpile(circuits, basis_gates=["cx", "rz", "sx", "x", "id"], optimization_level=optimization_level)
    compiled_circuits = transpile(compiled_circuits, backend, optimization_level=optimization_level)
    # else:
    #     compiled_circuits = circuits

    if execute:
        results = backend.run(compiled_circuits, shots=nshots)
    else:
        results = compiled_circuits

        
    print(f' Depth:', [circ.depth() for circ in compiled_circuits])
    print(f' 1q gates:', [get_num_gates(circ)[0] for circ in compiled_circuits])
    print(f' 2q gates:', [get_num_gates(circ)[1] for circ in compiled_circuits])

    if execute and counts:
        counts_list = [result.get_counts() for result in results]
        #qibo_counts = [counts_to_qibo(counts) for counts in counts_list]
        return counts_list, compiled_circuits
    elif execute:
        return results, compiled_circuits
    
    return compiled_circuits

