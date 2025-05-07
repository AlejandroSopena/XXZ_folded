from collections import Counter
import qnexus as qnx
from datetime import datetime


def counts_to_qibo(counts):
    new_counts = Counter()
    for key, value in counts.items():
        new_key = ''
        for i in range(len(key)):
            new_key += str(key[i])
        new_counts[new_key] = value
    return new_counts

def compile_quantinuum(circuits, name_project, optimization_level, nshots, device, counts=False):
    #qnx.login_with_credentials()

    if device == 'local_noiseless_simulator':
        from pytket.extensions.qiskit import AerBackend
        aer_state_b = AerBackend()
        circuits = aer_state_b.get_compiled_circuits(circuits)

        results_handle = aer_state_b.process_circuits(circuits)
        results = aer_state_b.get_results(results_handle)
    else:
        my_project_ref = qnx.projects.get_or_create(name=name_project)

        backend_config = qnx.QuantinuumConfig(device_name=device)

        circuit_refs = [qnx.circuits.upload(name=f"circuit_{i}",circuit=circuits[i],project=my_project_ref) for i in range(len(circuits))]

        compile_job = qnx.start_compile_job(circuits=circuit_refs, name=f"compilation_{name_project}_{datetime.now()}", 
                                            optimisation_level=optimization_level, backend_config=backend_config, project=my_project_ref)
        qnx.jobs.wait_for(compile_job)
        compiled_circuits = [item.get_output() for item in qnx.jobs.results(compile_job)]

        # print('Depth quantinuum:')
        # for i, circ in enumerate(compiled_circuits):
        #     print(f'Circuit {i}:')
        #     print(f' Depth:', circ.depth())
        #     print(f' 1q gates:', circ.n_1qb_gates())
        #     print(f' 2q gates:', circ.n_2qb_gates())

        execute_job_ref = qnx.start_execute_job(circuits=compiled_circuits, name=f"execution_{name_project}_{datetime.now()}", n_shots=[nshots] * len(compiled_circuits), 
                                                backend_config=backend_config, project=my_project_ref)
        print(execute_job_ref.df())
        qnx.jobs.wait_for(execute_job_ref)

        execute_job_result_refs = qnx.jobs.results(execute_job_ref)
        results = [execute_job_result_refs[i].download_result() for i in range(len(execute_job_result_refs))]

    if counts:
        counts_list = [result.get_counts() for result in results]
        qibo_counts = [counts_to_qibo(counts) for counts in counts_list]
        return qibo_counts
    
    return results
