import numpy as np

from qibo import gates, Circuit
from qibo.symbols import X, Y, Z
from qibo.hamiltonians import SymbolicHamiltonian, Hamiltonian
from qibo.quantum_info import fidelity, partial_trace
from qibo.backends import _check_backend, construct_backend

from initial_b_matrix import get_b_circuit
from XX_model import XX_model
from utils_quantinuum import counts_to_qibo, compile_quantinuum
from utils_ionq import compile_ionq
from qibo.symbols import Symbol


def partial_trace_vector(state, keep_indices):
    dim = len(state)
    num_qubits = int(np.log2(dim))

    traced_out = [i for i in range(num_qubits) if i not in keep_indices]

    dim_traced_out = 2**len(traced_out)
    dim_remaining = dim // dim_traced_out

    state_reshaped = state.reshape([2] * num_qubits)

    perm = keep_indices + traced_out
    state_permuted = np.transpose(state_reshaped, perm)

    state_permuted = state_permuted.reshape([dim_remaining, dim_traced_out])

    partial_trace_rho = np.einsum(
        'ij,kj->ik', state_permuted, state_permuted.conj())

    return partial_trace_rho


def partial_trace(rho, keep_indices):
    dim = rho.shape[0]
    num_qubits = int(np.log2(dim))

    traced_out = [i for i in range(num_qubits) if i not in keep_indices]

    dim_traced_out = 2**len(traced_out)
    dim_remaining = dim // dim_traced_out

    rho = np.reshape(rho, [2] * num_qubits * 2)
    perm = keep_indices + [num_qubits + i for i in keep_indices] + \
        traced_out + [num_qubits + i for i in traced_out]

    rho = np.transpose(rho, perm)

    new_shape = [dim_remaining, dim_remaining, dim_traced_out, dim_traced_out]
    rho = np.reshape(rho, new_shape)

    result = np.trace(rho, axis1=2, axis2=3)
    del rho

    return result


class XXZ_folded:
    """
    A class to build the circuits to prepare the eigenstates of the XXZ folded with two domain walls.

    Args:
        N (int): The number of bulk qubits.
        M (int): The number of domain walls.
        domain_pos (list): The positions of the domain walls.
    """
    def __init__(self, N=8, M=1, D=2, momentum_ints=[], domain_pos=[[5, 6, 7]], backend=None):
        self.N = N
        self.M = M
        self.D = D
        self.momentum_ints = momentum_ints
        self.domain_pos = domain_pos
        self.backend = _check_backend(backend)

    def _get_roots(self):
        roots = []
        if self.momentum_ints == []:
            self.momentum_ints = [i+1 for i in range(self.M)]
        for i in range(self.M):
            p = self.momentum_ints[i]*np.pi/(self.N+2-self.M-self.D)
            roots.append(p)
        self.roots = roots

    def get_b_circuit(self):
        b_circuit = get_b_circuit(self.N-self.D-self.M+1, self.M, self.roots, self.backend)

        return b_circuit

    def get_xx_b_circuit(self):
        # XX state N-D-M+1 qubits
        xx_model = XX_model(self.N-self.D-self.M+1, 2*self.M)
        roots1 = []
        for i in range(self.M):
            roots1.append(self.roots[i])
            roots1.append(-self.roots[i])

        xx_model.get_roots(roots1, 'Momentum')
        xx_model.P_list()
        xx_model.get_circuit()
        circ_xx = xx_model.circuit
        self.circ_xx = circ_xx

        circ_xx1 = Circuit(**circ_xx.init_kwargs)
        for gate in circ_xx.queue:
            if isinstance(gate, gates.X) == False:
                circ_xx1.add(gate)

        # B circuit + XX
        b_circuit = self.get_b_circuit()[0]
        circ_xxb = Circuit(self.N-self.D-self.M+1)
        circ_xxb.add(b_circuit.on_qubits(*range(2*self.M)))
        circ_xxb.add(circ_xx1.on_qubits(*range(self.N-self.D-self.M+1)))
        self.circ_xxb = circ_xxb

        return circ_xx, circ_xxb

    def get_U0_circ(self):
        # U_0 to produce magnon state |Psi_{M,0}_{N-D}>
        aux_qubits1 = self.M + 1
        circ_u0 = Circuit(self.N-self.D + aux_qubits1)
        circ_u0.add(gates.X(self.N-self.D))
        ii = 0
        jj = 1
        for q in reversed(range(self.N-self.D-self.M+1)):

            for j in range(self.M-1+ii,self.M):
                circ_u0.add(gates.SWAP(self.N-self.D + aux_qubits1 - 2 - j,
                            self.N-self.D + aux_qubits1 - 1 - j).controlled_by(q))
            if self.M-1+ii >= 1:
                ii -= 1

            for j in range(0, jj):#range(self.M-2+jj, self.M-1): #ii+1
                if q < q+self.M-1-j:
                    circ_u0.add(gates.SWAP(q, q+self.M-1 -
                                j).controlled_by(self.N-self.D+1+j))
            if jj < self.M-1:
                jj += 1
        self.circ_u0 = circ_u0

        return circ_u0    

    def move_before(self, m):
        if m==1:
            circ = Circuit(10)
            circ.add(gates.CNOT(1,4))
            # circ.add(gates.X(0))
            # circ.add(gates.TOFFOLI(0,1,4))
            # circ.add(gates.X(0))
            circ.add(gates.X(1))
            circ.add(gates.TOFFOLI(1,2,4))
            circ.add(gates.X(1))
            circ.add(gates.TOFFOLI(4,6,5))
            circ.add(gates.CNOT(5,0))
            circ.add(gates.CNOT(5,1))
            circ.add(gates.CNOT(5,7))
            circ.add(gates.SWAP(8,9).controlled_by(5))
            # circ.add(gates.SWAP(10,11).controlled_by(4)) #4
            circ.add(gates.TOFFOLI(4,6,5))
            circ.add(gates.CNOT(0,4)) 
            circ.add(gates.X(2))
            circ.add(gates.TOFFOLI(1,2,4))
            circ.add(gates.X(2))
            circ.add(gates.X(3))
            circ.add(gates.TOFFOLI(2,3,4))
            circ.add(gates.X(3))           
        else:
            if m > int(self.D/2):
                m = int(self.D/2)
            circ = Circuit(9 + m + 1)
            circ.add(gates.X(2))
            circ.add(gates.TOFFOLI(2,3,5))
            circ.add(gates.X(2))
            circ.add(gates.TOFFOLI(5,7,6))
            circ.add(gates.CNOT(6,1))
            circ.add(gates.CNOT(6,2))
            circ.add(gates.CNOT(6,8))
            for i in reversed(range(1, m+1)):
                circ.add(gates.SWAP(9+i-1, 9+i).controlled_by(6))
            # for i in reversed(range(1, int(self.D/2)+1)):
            #     circ.add(gates.SWAP(9+int(self.D/2+1)+i-1, 9+int(self.D/2+1)+i).controlled_by(5)) 
            circ.add(gates.TOFFOLI(5,7,6))
            circ.add(gates.X(0))
            circ.add(gates.TOFFOLI(0,1,5))
            circ.add(gates.X(0))
            circ.add(gates.X(4))
            circ.add(gates.TOFFOLI(3,4,5))
            circ.add(gates.X(4))
            circ.add(gates.TOFFOLI(5,7,6))
            circ.add(gates.CNOT(6,2))
            circ.add(gates.CNOT(6,3))
            circ.add(gates.CNOT(6,8))
            circ.add(gates.TOFFOLI(5,7,6))
            circ.add(gates.X(2))
            circ.add(gates.TOFFOLI(1,2,5))
            circ.add(gates.X(2))

        return circ
    

    def move_after(self, n_p):

        circ = Circuit(9 + n_p)
        circ.add(gates.X(2))
        circ.add(gates.TOFFOLI(2,3,5))
        circ.add(gates.X(2))
        circ.add(gates.X(8))
        circ.add(gates.TOFFOLI(5,8,6))
        circ.add(gates.CNOT(6,1))
        circ.add(gates.CNOT(6,2))
        circ.add(gates.CNOT(6,7))
        for i in reversed(range(1, n_p)):
            circ.add(gates.SWAP(9+i-1, 9+i).controlled_by(6))
        circ.add(gates.TOFFOLI(5,8,6))
        circ.add(gates.X(0))
        circ.add(gates.TOFFOLI(0,1,5))
        circ.add(gates.X(0))
        circ.add(gates.X(4))
        circ.add(gates.TOFFOLI(3,4,5))
        circ.add(gates.X(4))
        circ.add(gates.TOFFOLI(5,8,6))
        circ.add(gates.CNOT(6,2))
        circ.add(gates.CNOT(6,3))
        circ.add(gates.CNOT(6,7))
        circ.add(gates.TOFFOLI(5,8,6))
        circ.add(gates.X(8))
        circ.add(gates.X(2))
        circ.add(gates.TOFFOLI(1,2,5))
        circ.add(gates.X(2))

        return circ
    
    def p_scan_w(self,num_scan,qubit_0=False):
        np = int(self.D/2) + 1
        circ = Circuit(np+num_scan+1)
        for qq in reversed(range(np+1,np+num_scan)):
            circ.add(gates.X(qq-1))
            circ.add(gates.TOFFOLI(qq-1,qq,np+num_scan))
            circ.add(gates.X(qq-1))
        if qubit_0:
            circ.add(gates.CNOT(np,np+num_scan))
        for i in reversed(range(1, np)):
            circ.add(gates.SWAP(i-1, i).controlled_by(np+num_scan))
        if qubit_0:
            circ.add(gates.CNOT(np,np+num_scan))
        for qq in range(np,np+num_scan-1):
            circ.add(gates.X(qq))
            circ.add(gates.TOFFOLI(qq,qq+1,np+num_scan))
            circ.add(gates.X(qq))

        return circ
    
    def reset0_w(self, n=None):
        if n == 'last':
            circ = Circuit(7)
            circ.add(gates.X(1))
            circ.add(gates.TOFFOLI(1,2,3))
            circ.add(gates.TOFFOLI(0,3,5))
            circ.add(gates.CNOT(0,4))
            circ.add(gates.X(0))
            circ.add(gates.CNOT(0,4))
            circ.add(gates.X(0))
            circ.add(gates.TOFFOLI(3,4,6))
            circ.add(gates.CNOT(0,4))
            circ.add(gates.X(0))
            circ.add(gates.CNOT(0,4))
            circ.add(gates.X(0))
            circ.add(gates.TOFFOLI(1,2,3))
            circ.add(gates.X(1))
        else:
            circ = Circuit(8)
            circ.add(gates.X(1))
            circ.add(gates.TOFFOLI(1,2,4))
            circ.add(gates.TOFFOLI(0,4,6))
            circ.add(gates.CNOT(0,5))
            circ.add(gates.X(3))
            circ.add(gates.X(0))
            circ.add(gates.TOFFOLI(0,3,5))
            circ.add(gates.X(0))
            circ.add(gates.TOFFOLI(4,5,7))
            circ.add(gates.CNOT(0,5))
            circ.add(gates.X(0))
            circ.add(gates.TOFFOLI(0,3,5))
            circ.add(gates.X(0))
            circ.add(gates.X(3))
            circ.add(gates.TOFFOLI(1,2,4))
            circ.add(gates.X(1))

        return circ
    
    def reset_w(self, n=None):

        if n == 'last':
            circ = Circuit(9)
            circ.add(gates.X(2))
            circ.add(gates.TOFFOLI(2,3,4))
            circ.add(gates.TOFFOLI(1,4,6))
            circ.add(gates.TOFFOLI(0,6,7))
            circ.add(gates.X(1))
            circ.add(gates.CNOT(1,5))
            circ.add(gates.TOFFOLI(4,5,6))
            circ.add(gates.TOFFOLI(0,6,8))
            circ.add(gates.TOFFOLI(4,5,6))
            circ.add(gates.CNOT(1,5))
            circ.add(gates.X(1))
            circ.add(gates.TOFFOLI(1,4,6))
            circ.add(gates.TOFFOLI(2,3,4))
            circ.add(gates.X(2))
        elif n == 'first':
            circ = Circuit(9)
            circ.add(gates.X(1))
            circ.add(gates.TOFFOLI(1,2,4))
            circ.add(gates.TOFFOLI(0,6,7))
            circ.add(gates.X(3))
            circ.add(gates.CNOT(3,5))
            circ.add(gates.TOFFOLI(4,5,6))
            circ.add(gates.TOFFOLI(4,5,6))
            circ.add(gates.CNOT(3,5))
            circ.add(gates.X(3))
            circ.add(gates.TOFFOLI(1,2,4))
            circ.add(gates.X(1))           


        else:
            circ = Circuit(10)
            circ.add(gates.X(2))
            circ.add(gates.TOFFOLI(2,3,5))
            circ.add(gates.TOFFOLI(1,5,7))
            circ.add(gates.TOFFOLI(0,7,8))
            circ.add(gates.X(4))
            circ.add(gates.X(1))
            circ.add(gates.TOFFOLI(1,4,6))
            circ.add(gates.TOFFOLI(5,6,7))
            circ.add(gates.TOFFOLI(0,7,9))
            circ.add(gates.TOFFOLI(5,6,7))
            circ.add(gates.TOFFOLI(1,4,6))
            circ.add(gates.X(1))
            circ.add(gates.TOFFOLI(1,5,7))
            circ.add(gates.X(4))
            circ.add(gates.TOFFOLI(2,3,5))
            circ.add(gates.X(2))

        return circ

    def ip_scan_w(self,num_scan, qubit_0=False):
        np = int(self.D/2) + 1
        circ = Circuit(np+num_scan+1)
        for qq in reversed(range(np+1,np+num_scan)):
            circ.add(gates.X(qq-1))
            circ.add(gates.TOFFOLI(qq-1,qq,np+num_scan))
            circ.add(gates.X(qq-1))
        if qubit_0:
            circ.add(gates.CNOT(np,np+num_scan))
        for i in range(np-1):
            circ.add(gates.SWAP(i, i+1).controlled_by(np+num_scan))
        if qubit_0:
            circ.add(gates.CNOT(np,np+num_scan))
        for qq in range(np,np+num_scan-1):
            circ.add(gates.X(qq))
            circ.add(gates.TOFFOLI(qq,qq+1,np+num_scan))
            circ.add(gates.X(qq))

        return circ


    def get_D_circ_general(self):
        aux = 2 + (int(self.D/2) + 2) + (int(self.D/2)+1) # R0, Rc, Rr
        nqubits_d = 2*self.N - self.D
        r_0 = list(range(nqubits_d+aux-2*int(self.D/2)-2-2-1,nqubits_d+aux-2*int(self.D/2)-2-1)) 
        r_c = list(range(nqubits_d+aux-2*int(self.D/2)-2-1,nqubits_d+aux-int(self.D/2)-1))
        r_r = list(range(nqubits_d+aux-int(self.D/2)-1,nqubits_d+aux))
        #[phys,r0,rc,rr]
        circ_d = Circuit(nqubits_d+aux)
        circ_d.add(gates.X(r_c[1]))
        circ_d.add(gates.X(r_r[0]))
        nqubits = circ_d.nqubits


        index_domain = []
        k = 1
        for j in range(0, self.N - self.D):
            index_domain.append(k)
            k += 2
        k -= 1
        for j in range(self.N - self.D, self.N):
            index_domain.append(k)
            k += 1

        for domain in self.domain_pos:
            circ_d.add([gates.X(index_domain[index-1]) for index in domain])  # define domain

        index_p = []
        k = 0
        for j in range(0, self.N - self.D):
            index_p.append(k)
            k += 2
        #circ_d.add(gates.X(index_p[2]))
        #circ_d.add(gates.X(index_p[5])) # ADD MAGNON #
        for n in reversed(range(len(index_p))):
        #for n in [1]:  
            if n >= 1:
                #MOVE DOMAIN BEFORE

                circ_d.add(self.move_before(1).on_qubits(*[index_domain[0],index_domain[1], index_domain[2], index_domain[3], r_0[0], r_0[1], index_p[n], r_c[0], r_c[1], r_c[2]]))
                ii = 2
                for qq in range(n-1):
                    qubits = [index_domain[qq],index_domain[qq+1], index_domain[qq+2], index_domain[qq+3], index_domain[qq+4], r_0[0], r_0[1], index_p[n]] + r_c[0:ii+2]
                    circ_d.add(self.move_before(ii).on_qubits(*qubits))


                #MOVE DOMAIN AFTER
                for index, qq in enumerate(range(n-1,n+self.D-4,2)):
                    if index != 0:
                        circ_d.add(gates.X(r_c[0]))
                        circ_d.add(gates.TOFFOLI(r_c[0], r_c[2+index-1], r_c[1]))
                        circ_d.add(gates.X(r_c[0]))
                    circ_d.add(gates.TOFFOLI(r_c[0], r_c[2+index], r_c[1]))  
                    qubits = [index_domain[qq],index_domain[qq+1], index_domain[qq+2], index_domain[qq+3], index_domain[qq+4], r_0[0], r_0[1], r_c[0], r_c[1]] + r_c[(2+index)::]
                    circ_d.add(self.move_after(n_p=len(r_c[(2+index)::])).on_qubits(*qubits))
                    qq+=1
                    qubits = [index_domain[qq],index_domain[qq+1], index_domain[qq+2], index_domain[qq+3], index_domain[qq+4], r_0[0], r_0[1], r_c[0], r_c[1]] + r_c[(2+index)::]
                    circ_d.add(self.move_after(n_p=len(r_c[(2+index)::])).on_qubits(*qubits))

                if self.D > 2:
                    circ_d.add(gates.TOFFOLI(r_c[0], r_c[-2], r_c[1]))  
                for ii in reversed(range(2,len(r_c)-2)):
                    circ_d.add(gates.CNOT(r_c[ii], r_c[1]))

                circ_d.add(gates.X(index_domain[n+self.D-2]))
                circ_d.add(gates.TOFFOLI(index_domain[n+self.D-2],index_domain[n+self.D-1],r_0[0]))
                circ_d.add(gates.X(index_domain[n+self.D-2]))
                circ_d.add(gates.X(index_domain[n+self.D-3]))
                circ_d.add(gates.TOFFOLI(index_domain[n+self.D-3],index_domain[n+self.D-2],r_0[0]))
                circ_d.add(gates.X(index_domain[n+self.D-3]))

                circ_d.add(gates.X(index_domain[n+self.D]))
                circ_d.add(gates.TOFFOLI(index_domain[n+self.D-1],index_domain[n+self.D],r_0[0]))
                circ_d.add(gates.X(index_domain[n+self.D]))
                circ_d.add(gates.X(index_domain[n+self.D-1]))
                circ_d.add(gates.TOFFOLI(index_domain[n+self.D-2],index_domain[n+self.D-1],r_0[0]))
                circ_d.add(gates.X(index_domain[n+self.D-1]))


            #ADD MAGNON
            circ_d.add(gates.TOFFOLI(index_p[n], r_c[1], index_domain[n]))
            for i, ii in enumerate(index_domain[n+1:n+self.D:2]):
                circ_d.add(gates.TOFFOLI(r_c[0], r_c[i+2], ii))
                circ_d.add(gates.X(r_c[0]))
                circ_d.add(gates.TOFFOLI(r_c[0], r_c[i+2], index_domain[index_domain.index(ii)+1]))
                circ_d.add(gates.X(r_c[0]))

            # # # RESET

            #   RESET R_aux_n, second qubit Rc
            if n==0:
                circ_d.add(gates.CNOT(index_domain[n], r_0[0]))
            else:
                circ_d.add(gates.X(index_domain[n-1]))
                circ_d.add(gates.TOFFOLI(index_domain[n-1], index_domain[n], r_0[0]))
            circ_d.add(gates.X(index_domain[n+1]))
            circ_d.add(gates.TOFFOLI(index_domain[n+1], r_0[0], index_p[n]))
            circ_d.add(gates.X(index_domain[n+1]))
            if n==0:
                circ_d.add(gates.CNOT(index_domain[n], r_0[0]))
            else:
                circ_d.add(gates.TOFFOLI(index_domain[n-1], index_domain[n], r_0[0]))
                circ_d.add(gates.X(index_domain[n-1]))
            for ii in r_c[2::]:
                circ_d.add(gates.CNOT(ii, index_p[n]))
                circ_d.add(gates.CNOT(ii, r_c[1]))

            # # #   #RESET Rr 1

            if n>= 1:

                end = n - 1 + 1
                start = 0
                step = 4
                num_scans = int((end-start)/step)
                rest = int(end - step*num_scans)
                if rest > 0:
                    if num_scans == 0:
                        end_rest = end
                    else:
                        end_rest = end-step*num_scans
                    phys_q = [index_domain[j] for j in range(start,end_rest)]
                    q_r = r_r + phys_q + [r_0[0]]                    
                    if phys_q[0] == index_domain[0]:
                        qubit_0 = True
                    else:
                        qubit_0 = False
                    circ_d.add(self.p_scan_w(num_scan=(end_rest-start),qubit_0=qubit_0).on_qubits(*q_r))
                for i in range(num_scans):
                    if i == 0 and rest == 0:
                        phys_q = [index_domain[j] for j in range(end-step*num_scans+step*i,end-step*num_scans+step*(i+1))]
                        q_r = r_r + phys_q + [r_0[0]]
                        if phys_q[0] == index_domain[0]:
                            qubit_0 = True
                        else:
                            qubit_0 = False
                        circ_d.add(self.p_scan_w(num_scan=step,qubit_0=qubit_0).on_qubits(*q_r))
                    else:
                        phys_q = [index_domain[j] for j in range(end-step*num_scans-1+step*i,end-step*num_scans+step*(i+1))]
                        q_r = r_r + phys_q + [r_0[0]]
                        circ_d.add(self.p_scan_w(num_scan=step+1).on_qubits(*q_r))


            # #   #RESET Rc
            if n>= 1:
                num_scan = 3
                # if self.D > 2:
                #     if n == len(index_p)-1:
                #         circ_d.add(self.reset0_w(n='last').on_qubits(*[index_domain[n],index_domain[n+1], index_domain[n+2], r_0[0], r_0[1], r_c[0], r_c[2]]))
                #     else:      
                #         circ_d.add(self.reset0_w(n=None).on_qubits(*[index_domain[n],index_domain[n+1], index_domain[n+2], index_domain[n+3], r_0[0], r_0[1], r_c[0], r_c[2]]))
                #     q_r = r_r + [index_domain[n-1],index_domain[n], index_domain[n+1], r_0[0]]
                #     circ_d.add(self.p_scan_w(num_scan).on_qubits(*q_r))

                for index, qq in enumerate(range(n+2,n+self.D-1,2)):
                    q_r = [r_r[index+1], index_domain[qq-2],index_domain[qq-1], index_domain[qq], index_domain[qq+1]] + r_0 + [index_p[n]] + [r_c[0], r_c[2+index]]
                    circ_d.add(self.reset_w().on_qubits(*q_r))
                    q_r = r_r + [index_domain[qq-3],index_domain[qq-2], index_domain[qq-1], r_0[0]]
                    circ_d.add(self.p_scan_w(num_scan).on_qubits(*q_r))


                qq = n + self.D
                if n == len(index_p)-1:
                    q_r = [r_r[-1], index_domain[qq-2],index_domain[qq-1], index_domain[qq]] + r_0 + [index_p[n]] + [r_c[0], r_c[-1]]
                    circ_d.add(self.reset_w('last').on_qubits(*q_r))
                else:
                    q_r = [r_r[-1], index_domain[qq-2],index_domain[qq-1], index_domain[qq], index_domain[qq+1]] + r_0 + [index_p[n]] + [r_c[0], r_c[-1]]
                    circ_d.add(self.reset_w().on_qubits(*q_r))


            # # # #   #RESET Rr 2



            if n>= 1:

                end = n + self.D - 3 +1
                start = 0
                step = 4
                num_scans = int((end-start)/step)
                rest = int(end - step*num_scans)

                for i in reversed(range(num_scans)):
                    if i == 0 and rest == 0:
                        phys_q = [index_domain[j] for j in range(end-step*num_scans+step*i,end-step*num_scans+step*(i+1))]
                        q_r = r_r + phys_q + [r_0[0]]
                        if phys_q[0] == index_domain[0]:
                            qubit_0 = True
                        else:
                            qubit_0 = False
                        circ_d.add(self.ip_scan_w(num_scan=step,qubit_0=qubit_0).on_qubits(*q_r))
                    else:
                        phys_q = [index_domain[j] for j in range(end-step*num_scans-1+step*i,end-step*num_scans+step*(i+1))]
                        q_r = r_r + phys_q + [r_0[0]]
                        circ_d.add(self.ip_scan_w(num_scan=step+1).on_qubits(*q_r))

                if rest > 0:
                    if num_scans == 0:
                        end_rest = end
                    else:
                        end_rest = end-step*num_scans
                    phys_q = [index_domain[j] for j in range(start,end_rest)]
                    q_r = r_r + phys_q + [r_0[0]]
                    if phys_q[0] == index_domain[0]:
                        qubit_0 = True
                    else:
                        qubit_0 = False
                    circ_d.add(self.ip_scan_w(num_scan=(end_rest-start),qubit_0=qubit_0).on_qubits(*q_r))


        self.circ_d = circ_d
        # sym_state = circ_d().symbolic()
        # sym_state = sym_state[7:-1]
        # new_state = ''.join([sym_state[i] for i in index_domain])
        # print(new_state)
        # print('r_0',''.join([sym_state[i] for i in r_0]))
        # print('r_c',''.join([sym_state[i] for i in r_c]))
        # print('r_r',''.join([sym_state[i] for i in r_r]))
        # print('r_aux',''.join([sym_state[i] for i in index_p]))

        return circ_d

    # def get_D_circ_N5_M1(self):
    #     aux = 3
    #     nqubits_d = 2*self.N - self.D
    #     circ_d = Circuit(nqubits_d+aux)
    #     nqubits = circ_d.nqubits

    #     index_domain = []

    #     i = 1
    #     for j in self.domain_pos[0]:
    #         if j <= self.N - self.D:
    #             index_domain.append(2*j-1)
    #         else:
    #             index_domain.append(2*(self.N-self.D)-1+i)
    #             i += 1

    #     circ_d.add([gates.X(index) for index in index_domain])  # define domain

    #     index_p = []
    #     k = 0
    #     for j in range(0, self.N - self.D):
    #         index_p.append(k)
    #         k += 2

    #     index_domain = []
    #     k = 1
    #     for j in range(0, self.N - self.D):
    #         index_domain.append(k)
    #         k += 2
    #     k -= 1
    #     for j in range(self.N - self.D, self.N):
    #         index_domain.append(k)
    #         k += 1

    #     circ_d.add(gates.SWAP(index_p[2], nqubits-3))

    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
    #     circ_d.add(gates.CNOT(index_domain[0], nqubits-2))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
    #     circ_d.add(gates.X(index_domain[0]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[0], nqubits-3))
    #     circ_d.add(gates.X(index_domain[0]))

    #     circ_d.add(gates.X(index_domain[4]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[4], nqubits-3))
    #     circ_d.add(gates.X(index_domain[4]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(gates.X(index_domain[2]))

    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))     
    #     circ_d.add(gates.CNOT(nqubits-1, index_domain[3]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))     
    #     circ_d.add(gates.X(index_domain[2]))

    #     circ_d.add(gates.X(index_domain[3]))
    #     circ_d.add(gates.CNOT(index_domain[3], nqubits-3))
    #     circ_d.add(gates.CNOT(index_domain[3], nqubits-1))
    #     circ_d.add(gates.X(index_domain[3]))

    #     circ_d.add(gates.SWAP(index_p[1], nqubits-3))

    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[0], nqubits-3))

    #     circ_d.add(gates.CNOT(nqubits-3, index_domain[1]))
    #     circ_d.add(gates.CNOT(nqubits-1, index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-1, index_domain[1]))

    #     circ_d.add(gates.X(index_domain[0]))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(gates.X(
    #         nqubits-3).controlled_by(index_domain[0], index_domain[1], index_domain[2]))
    #     circ_d.add(gates.X(index_domain[0]))
    #     circ_d.add(gates.X(nqubits-3).controlled_by(
    #         index_domain[0], index_domain[1], index_domain[2], index_domain[3]))
    #     circ_d.add(gates.X(nqubits-1).controlled_by(
    #         index_domain[0], index_domain[1], index_domain[2], index_domain[3]))
    #     circ_d.add(gates.X(index_domain[2]))

    #     circ_d.add(gates.SWAP(index_p[0], nqubits-3))

    #     circ_d.add(gates.CNOT(nqubits-3, index_domain[0]))

    #     circ_d.add(gates.X(index_domain[1]))
    #     circ_d.add(
    #         gates.X(nqubits-3).controlled_by(index_domain[0], index_domain[1]))
    #     circ_d.add(gates.X(index_domain[1]))

    #     self.circ_d = circ_d

    #     return circ_d
    
    def get_D_circ_N5_M1(self):
        aux = 3
        nqubits_d = 2*self.N - self.D
        circ_d = Circuit(nqubits_d+aux)
        nqubits = circ_d.nqubits

        index_domain = []

        i = 1
        for j in self.domain_pos[0]:
            if j <= self.N - self.D:
                index_domain.append(2*j-1)
            else:
                index_domain.append(2*(self.N-self.D)-1+i)
                i += 1

        circ_d.add([gates.X(index) for index in index_domain])  # define domain

        index_p = []
        k = 0
        for j in range(0, self.N - self.D):
            index_p.append(k)
            k += 2

        index_domain = []
        k = 1
        for j in range(0, self.N - self.D):
            index_domain.append(k)
            k += 2
        k -= 1
        for j in range(self.N - self.D, self.N):
            index_domain.append(k)
            k += 1

        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[2],nqubits-3))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_p[2],nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[2],nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_p[2],nqubits-3))
        circ_d.add(gates.X(index_domain[0]))

        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.TOFFOLI(index_domain[4],index_p[2],nqubits-3))
        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-1))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[2],nqubits-3))
        circ_d.add(gates.X(index_domain[2]))

        circ_d.add(gates.CNOT(nqubits-2,index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-1,index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-1,index_domain[4]))

        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.CNOT(index_domain[3],index_p[2]))
        circ_d.add(gates.CNOT(index_domain[3],nqubits-2))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_domain[3],nqubits-1))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.X(index_domain[3]))

        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[1],nqubits-3))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_p[1],nqubits-3))

        circ_d.add(gates.CNOT(index_p[1],index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-2,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-2,index_domain[2]))

        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],nqubits-3,index_p[1]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],nqubits-3))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[3],index_p[1]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[3],nqubits-2))

        circ_d.add(gates.CNOT(index_p[0],index_domain[0]))

        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],index_p[0]))
        circ_d.add(gates.X(index_domain[1]))


        self.circ_d = circ_d

        # sym_state = circ_d().symbolic()
        # sym_state = sym_state[7:-1]
        # new_state = ''.join([sym_state[i] for i in index_domain])
        # print(new_state)
        # print('aux',''.join([sym_state[i] for i in [nqubits-3,nqubits-2,nqubits-1]]))
        # print('r_magnon',''.join([sym_state[i] for i in index_p]))

        return circ_d

    # def get_D_circ_N6_M1(self):
    #     aux = 3
    #     nqubits_d = 2*self.N - self.D
    #     circ_d = Circuit(nqubits_d+aux)
    #     nqubits = circ_d.nqubits

    #     index_domain = []

    #     i = 1
    #     for j in self.domain_pos[0]:
    #         if j <= self.N - self.D:
    #             index_domain.append(2*j-1)
    #         else:
    #             index_domain.append(2*(self.N-self.D)-1+i)
    #             i += 1

    #     circ_d.add([gates.X(index) for index in index_domain])  # define domain

    #     index_p = []
    #     k = 0
    #     for j in range(0, self.N - self.D):
    #         index_p.append(k)
    #         k += 2

    #     index_domain = []
    #     k = 1
    #     for j in range(0, self.N - self.D):
    #         index_domain.append(k)
    #         k += 2
    #     k -= 1
    #     for j in range(self.N - self.D, self.N):
    #         index_domain.append(k)
    #         k += 1

    #     #circ_d.add(gates.X(index_p[1])) # ADD MAGNON

    #     circ_d.add(gates.SWAP(index_p[3], nqubits-3))

    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
    #     circ_d.add(gates.CNOT(index_domain[0], nqubits-2))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[2], index_domain[3], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], index_domain[3], nqubits-3))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
    #     circ_d.add(gates.X(index_domain[0]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[0], index_domain[1], nqubits-3))
    #     circ_d.add(gates.X(index_domain[0]))
    #     circ_d.add(gates.X(index_domain[3]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[3], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[3], nqubits-3))
    #     circ_d.add(gates.X(index_domain[3]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
    #     circ_d.add(gates.X(index_domain[1]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[1], nqubits-3))
    #     circ_d.add(gates.X(index_domain[1]))

    #     circ_d.add(gates.X(index_domain[4]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[4], nqubits-3))
    #     circ_d.add(gates.X(index_domain[4]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(gates.X(index_domain[5]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[4], index_domain[5], nqubits-3))
    #     circ_d.add(gates.X(index_domain[5]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
    #     circ_d.add(gates.X(index_domain[3]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], index_domain[3]))
    #     circ_d.add(gates.X(index_domain[3]))

    #     circ_d.add(gates.X(index_domain[3]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[3], nqubits-3))
    #     circ_d.add(gates.CNOT(nqubits-1, index_domain[4]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[5]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[3], nqubits-3))
    #     circ_d.add(gates.X(index_domain[3]))

    #     circ_d.add(gates.X(index_domain[4]))
    #     circ_d.add(gates.X(
    #         nqubits-3).controlled_by(index_domain[4], index_domain[5]))
    #     circ_d.add(gates.X(
    #         nqubits-1).controlled_by(index_domain[4], index_domain[5]))
    #     circ_d.add(gates.X(index_domain[4]))

    #     circ_d.add(gates.SWAP(index_p[2], nqubits-3))

    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
    #     #circ_d.add(gates.CNOT(index_domain[0],nqubits-2))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[0], nqubits-3)) #NEW
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[2], index_domain[3], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], index_domain[3], nqubits-3))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
    #     circ_d.add(gates.X(index_domain[0]))
    #     circ_d.add( #NEW
    #         gates.X(nqubits-2).controlled_by(index_domain[0], index_domain[1], nqubits-3)) #NEW
    #     circ_d.add(gates.X(index_domain[0]))

    #     circ_d.add(gates.X(index_domain[4]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[4], nqubits-3))
    #     circ_d.add(gates.X(index_domain[4]))
    #     circ_d.add(gates.CNOT(nqubits-3, index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-1, index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-1, index_domain[3]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
    #     # circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
    #     # circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
    #     # circ_d.add(gates.X(index_domain[2]))
    #     # circ_d.add(
    #     #     gates.X(nqubits-2).controlled_by(index_domain[1], index_domain[2], nqubits-3))
        
    #     # circ_d.add(
    #     #     gates.X(nqubits-2).controlled_by(index_domain[1], index_domain[2], nqubits-3))
    #     # circ_d.add(gates.X(index_domain[2]))
    #     # circ_d.add(gates.CNOT(nqubits-3, index_domain[2]))
    #     # circ_d.add(gates.CNOT(nqubits-1, index_domain[2]))
    #     # circ_d.add(gates.CNOT(nqubits-1, index_domain[3]))
    #     # circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
    #     # circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[1], index_domain[2], nqubits-3))
    #     circ_d.add(gates.X(index_domain[2]))


    #     circ_d.add(gates.X(index_domain[3]))
    #     circ_d.add(gates.X(index_domain[1]))
    #     circ_d.add(gates.X(
    #         nqubits-3).controlled_by(index_domain[1], index_domain[2], index_domain[3]))
    #     circ_d.add(gates.X(index_domain[1]))
    #     circ_d.add(gates.X(index_domain[3]))

    #     circ_d.add(gates.X(
    #         nqubits-3).controlled_by(index_domain[1], index_domain[4]))
    #     circ_d.add(gates.X(
    #         nqubits-1).controlled_by(index_domain[1], index_domain[4]))



      
    #     # circ_d.add(gates.X(index_domain[3]))
    #     # circ_d.add(gates.X(index_domain[1]))
    #     # circ_d.add(gates.X(
    #     #     nqubits-3).controlled_by(index_domain[1], index_domain[2], index_domain[3]))
    #     # circ_d.add(gates.X(index_domain[3]))
    #     # circ_d.add(gates.X(index_domain[1]))
    #     # # circ_d.add(
    #     # #     gates.X(nqubits-1).controlled_by(index_domain[1], index_domain[4]))
    #     # # circ_d.add(
    #     # #     gates.X(nqubits-3).controlled_by(index_domain[1], index_domain[4]))
    #     # circ_d.add(gates.CNOT(index_domain[1], nqubits-1))
    #     # circ_d.add(gates.CNOT(index_domain[1], nqubits-3)) #HERE FAIL




    #     circ_d.add(gates.SWAP(index_p[1], nqubits-3))

    #     circ_d.add(
    #         gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
    #     circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
    #     circ_d.add(
    #         gates.X(nqubits-2).controlled_by(index_domain[0], nqubits-3))

    #     circ_d.add(gates.CNOT(nqubits-3, index_domain[1]))
    #     circ_d.add(gates.CNOT(nqubits-1, index_domain[2]))
    #     circ_d.add(gates.CNOT(nqubits-1, index_domain[1]))

    #     circ_d.add(gates.X(index_domain[2]))
    #     circ_d.add(gates.X(index_domain[0]))
    #     circ_d.add(gates.X(
    #         nqubits-3).controlled_by(index_domain[0], index_domain[1], index_domain[2]))
    #     circ_d.add(gates.X(index_domain[0]))
    #     circ_d.add(gates.X(
    #         nqubits-3).controlled_by(index_domain[0], index_domain[2], index_domain[3]))
    #     circ_d.add(gates.X(
    #         nqubits-1).controlled_by(index_domain[0], index_domain[2], index_domain[3]))
    #     circ_d.add(gates.X(index_domain[2]))

    #     circ_d.add(gates.SWAP(index_p[0], nqubits-3))

    #     circ_d.add(gates.CNOT(nqubits-3, index_domain[0]))

    #     circ_d.add(gates.X(index_domain[1]))
    #     circ_d.add(
    #         gates.X(nqubits-3).controlled_by(index_domain[0], index_domain[1]))
    #     circ_d.add(gates.X(index_domain[1]))

    #     self.circ_d = circ_d

    #     return circ_d

    def get_D_circ_N6_M1(self):
        aux = 3
        nqubits_d = 2*self.N - self.D
        circ_d = Circuit(nqubits_d+aux)
        nqubits = circ_d.nqubits

        index_domain = []

        i = 1
        for j in self.domain_pos[0]:
            if j <= self.N - self.D:
                index_domain.append(2*j-1)
            else:
                index_domain.append(2*(self.N-self.D)-1+i)
                i += 1

        circ_d.add([gates.X(index) for index in index_domain])  # define domain

        index_p = []
        k = 0
        for j in range(0, self.N - self.D):
            index_p.append(k)
            k += 2

        index_domain = []
        k = 1
        for j in range(0, self.N - self.D):
            index_domain.append(k)
            k += 2
        k -= 1
        for j in range(self.N - self.D, self.N):
            index_domain.append(k)
            k += 1

        #insert n=3

        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[3],nqubits-3))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.CNOT(index_domain[0],nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_domain[3],nqubits-1))
        circ_d.add(gates.TOFFOLI(index_p[3],nqubits-1,nqubits-3))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_domain[3],nqubits-1))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],nqubits-1))
        circ_d.add(gates.TOFFOLI(index_p[3],nqubits-1,nqubits-3))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],nqubits-1))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.TOFFOLI(index_domain[3],index_p[3],nqubits-3))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(gates.TOFFOLI(index_domain[1],index_p[3],nqubits-3))
        circ_d.add(gates.X(index_domain[1]))

        circ_d.add(gates.X(index_domain[4]))   
        circ_d.add(gates.TOFFOLI(index_domain[4],index_p[3],nqubits-3))
        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[3]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[3],nqubits-3))
        circ_d.add(gates.X(index_domain[2])) 
        circ_d.add(gates.X(index_domain[5]))
        circ_d.add(gates.TOFFOLI(index_domain[4],index_domain[5],nqubits-1))
        circ_d.add(gates.TOFFOLI(index_p[3],nqubits-1,nqubits-3))
        circ_d.add(gates.TOFFOLI(index_domain[4],index_domain[5],nqubits-1))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[4]))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_domain[3],nqubits-3))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.TOFFOLI(index_domain[1],index_domain[5],nqubits-1))
        circ_d.add(gates.X(index_domain[5]))

        circ_d.add(gates.CNOT(nqubits-2,index_domain[4]))  
        circ_d.add(gates.CNOT(nqubits-1,index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-1,index_domain[5]))


        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.TOFFOLI(index_domain[3],index_p[3],nqubits-1))
        circ_d.add(gates.X(index_domain[3]))   
        circ_d.add(gates.CNOT(nqubits-2,index_p[3]))
        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.TOFFOLI(index_domain[4],index_domain[5],nqubits-2))
        circ_d.add(gates.X(index_domain[4]))   

        # insert n=2

        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[2],nqubits-3))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_p[2],nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_domain[3],index_p[3]))
        circ_d.add(gates.TOFFOLI(index_p[2],index_p[3],nqubits-3))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_domain[3],index_p[3]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],index_p[3]))
        circ_d.add(gates.TOFFOLI(index_p[2],index_p[3],nqubits-3))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],index_p[3]))
        circ_d.add(gates.X(index_domain[0]))
       
        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.TOFFOLI(index_domain[4],index_p[2],nqubits-3))
        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-1))
        circ_d.add(gates.X(index_domain[2])) 
        circ_d.add(gates.TOFFOLI(index_domain[1],index_domain[2],index_p[3]))
        circ_d.add(gates.TOFFOLI(index_p[2],index_p[3],nqubits-3))
        circ_d.add(gates.TOFFOLI(index_domain[1],index_domain[2],index_p[3]))
        circ_d.add(gates.X(index_domain[2]))

        circ_d.add(gates.CNOT(index_p[2],index_domain[2]))  
        circ_d.add(gates.CNOT(nqubits-2,index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-2,index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-1,index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-1,index_domain[4]))

        circ_d.add(gates.X(index_domain[1]))  
        circ_d.add(gates.TOFFOLI(index_domain[1],index_domain[2],nqubits-3))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.TOFFOLI(index_domain[3],nqubits-3,index_p[2]))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.TOFFOLI(index_domain[1],index_domain[2],nqubits-3))
        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[2],nqubits-1))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-2,index_p[2]))
        circ_d.add(gates.TOFFOLI(index_domain[1],index_domain[4],nqubits-2))

        # # insert n=1

        circ_d.add(gates.TOFFOLI(index_domain[2],index_p[1],nqubits-3))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-3,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-3,nqubits-2))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_p[1],nqubits-3))

        circ_d.add(gates.CNOT(index_p[1],index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-2,index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-2,index_domain[2]))

        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],nqubits-3,index_p[1]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],nqubits-3))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-2,index_p[1]))  
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[3],nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[2],nqubits-3,nqubits-2))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[3],nqubits-3))

        # insert n=0

        circ_d.add(gates.CNOT(index_p[0],index_domain[0]))

        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(gates.TOFFOLI(index_domain[0],index_domain[1],index_p[0]))
        circ_d.add(gates.X(index_domain[1]))

        self.circ_d = circ_d

        # sym_state = circ_d().symbolic()
        # sym_state = sym_state[7:-1]
        # new_state = ''.join([sym_state[i] for i in index_domain])
        # print(new_state)
        # print('aux',''.join([sym_state[i] for i in [nqubits-3,nqubits-2,nqubits-1]]))
        # print('r_magnon',''.join([sym_state[i] for i in index_p]))

        return circ_d
    
    def get_D_circ(self):
        if self.N == 5 and self.M == 1 and self.D == 2:
            return self.get_D_circ_N5_M1()
        elif self.N == 6 and self.M == 1 and self.D == 2:
            return self.get_D_circ_N6_M1()
        else:
            return self.get_D_circ_general()

    def get_Psi_M_0_circ(self):
        # |Psi_{M,0}_{N-D}> = U_0 (|phi_xx>_{M,N-D-M+1} \otimes |0>^{M-1})
        if self.M == 1:
            aux_qubits1 = 0
        else:
            aux_qubits1 = self.M + 1
        circ = Circuit(self.N-self.D + aux_qubits1)
        circ.add(self.circ_xxb.on_qubits(*range(self.N-self.D-self.M+1)))
        if self.M > 1:
            circ.add(self.circ_u0.on_qubits(
                *range(self.N-self.D + aux_qubits1)))
        self.circ_Psi_M_0 = circ

        return circ

    def get_full_circ(self):
        # |Psi_{M,0}_{N-D}> X |0>^D

        if self.M == 1:
            aux_qubits1 = 0
        else:
            aux_qubits1 = self.M + 1

        circ1 = self.circ_Psi_M_0

        if self.D == 0:
            self.circ_full = circ1
            return circ1

        # |Psi_{M,D}>
        aux = 4
        if self.N == 5 and self.M == 1 and self.D == 2:
            aux = 3
        elif self.N == 6 and self.M == 1 and self.D == 2:
            aux = 3
        else:
            aux = 2 + int(self.D/2) + 2 + int(self.D/2) + 1
        final_aux = max(aux, aux_qubits1)
        circ_full = Circuit(2*self.N-self.D+final_aux)

        index_p = []
        k = 0
        for j in range(0, self.N - self.D):
            index_p.append(k)
            k += 2

        new_qubits_1 = index_p + \
            list(range(2*self.N-self.D, 2*self.N-self.D+aux_qubits1))
        new_qubits_2 = list(range(0, 2*self.N-self.D)) + list(range(2 *
                                                                    self.N-self.D, 2*self.N-self.D
                                                                    +aux))

        circ_full.add(circ1.on_qubits(*new_qubits_1))
        if aux_qubits1 > 0:
            circ_full.add(gates.X(new_qubits_1[-1]))
        circ_full.add(self.circ_d.on_qubits(*new_qubits_2))

        self.circ_full = circ_full

        return circ_full

    def get_xxz_folded_hamiltonian(self, boundaries=True):
        ham = 0
        if boundaries:
            for j in range(self.N-1):
                ham += -(1/8)*(1+Z(j)*Z(j+3))*(X(j+1)*X(j+2)+Y(j+1)*Y(j+2))
        else:
            for j in range(0, self.N-3):
                ham += -(1/8)*(1+Z(j)*Z(j+3))*(X(j+1)*X(j+2)+Y(j+1)*Y(j+2))
            j = 0
            ham += -(1/8)*(1+Z(j+2))*(X(j)*X(j+1)+Y(j)*Y(j+1))
            j = self.N-3
            ham += -(1/8)*(1+Z(j))*(X(j+1)*X(j+2)+Y(j+1)*Y(j+2))
        ham = SymbolicHamiltonian(ham, backend=self.backend)

        return ham

    def get_q1(self, boundaries=True):
        q1 = 0
        if boundaries:
            for j in range(self.N+2):
                q1 += (1/2)*(1-Z(j))
        else:
            for j in range(0, self.N):
                q1 += (1/2)*(1-Z(j))
        q1 = SymbolicHamiltonian(q1, backend=self.backend)

        return q1

    def get_q2(self, boundaries=True):
        q2 = 0
        if boundaries:
            for j in range(self.N+1):
                q2 += (1/2)*(1-Z(j)*Z(j+1))
        else:
            for j in range(0, self.N-1):
                q2 += (1/2)*(1-Z(j)*Z(j+1))

            q2 += (1/2)*(1-Z(0))
            q2 += (1/2)*(1-Z(self.N-1))

        q2 = SymbolicHamiltonian(q2, backend=self.backend)

        return q2
    
    def get_ej(self, j, boundaries=True):
        ej = 1    
        if boundaries: 
            for k in range(self.N-j+1, self.N+1):
                ej *= (1+Z(k))/2
        else:
            for k in range(self.N-j, self.N):
                ej *= (1+Z(k))/2

        ej = SymbolicHamiltonian(ej, backend=self.backend)

        return ej
    
    def get_nonlocal_pauli(self, num, boundaries=True):
        if boundaries:
            insertion_index = np.ceil(np.linspace(0,self.N+1,num))
        else:
            insertion_index = np.ceil(np.linspace(0,self.N-1,num))

        obs = 1
        for i in insertion_index:
            obs *= Z(int(i))

        obs = SymbolicHamiltonian(obs, backend=self.backend)

        return obs
    
    def get_state(self, noise_model=None, boundaries=True, density_matrix=False, state=None, layout=None):
        # if self.D == 0:
        #     circ = self.circ_Psi_M_0
        # else:
        circ = self.circ_full
        if noise_model is not None:
            circ = noise_model.apply(circ)
        circ.density_matrix = density_matrix
        if state is None:
            result = self.backend.execute_circuit(circ)  # circ()
            state1 = result.state()
        else:
            state1 = state

        if self.D != 0:
            if boundaries:
                if (self.N == 5 and self.M == 1 and self.D == 2) or (self.N == 6 and self.M == 1 and self.D == 2):
                    keep = [self.circ_full.nqubits-1]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-2]
                else:
                    keep = [self.circ_full.nqubits-2-(int(self.D/2) + 2 + int(self.D/2) + 1)]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-1-(int(self.D/2) + 2 + int(self.D/2) + 1)]
            else:
                if self.M == 1:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
                else:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
        else:
            if boundaries:
                raise ValueError(
                    'Boundaries not implemented for D=0')
            else:
                keep = list(range(self.N))

        if layout is not None:
            keep = [layout[k] for k in keep]

        if density_matrix == True:
            state1 = partial_trace(state1, keep)
        else:
            state1 = partial_trace_vector(state1, keep)

        return state1

    def check_fidelity(self, state, boundaries=True):
        ham = self.get_xxz_folded_hamiltonian(boundaries).matrix
        q1 = self.get_q1(boundaries).matrix
        q2 = self.get_q2(boundaries).matrix
        if len(np.shape(state)) == 2:
            dm = True
        else:
            dm = False

        if dm:
            ham_state = ham@state@ham.conjugate().transpose()
            ham_state = ham_state / np.trace(ham_state)
        else:
            ham_state = ham@state
            ham_state = ham_state / np.linalg.norm(ham_state)
        fid_ham = fidelity(ham_state, state, backend=self.backend)

        if dm:
            q1_state = q1@state@q1.conjugate().transpose()
            q1_state = q1_state / np.trace(q1_state)
        else:
            q1_state = q1@state
            q1_state = q1_state / np.linalg.norm(q1_state)
        fid_q1 = fidelity(q1_state, state, backend=self.backend)

        if dm:
            q2_state = q2@state@q2.conjugate().transpose()
            q2_state = q2_state / np.trace(q2_state)
        else:
            q2_state = q2@state
            q2_state = q2_state / np.linalg.norm(q2_state)
        fid_q2 = fidelity(q2_state, state, backend=self.backend)

        return fid_ham, fid_q1, fid_q2

    def get_energy(self, state, boundaries):
        ham = self.get_xxz_folded_hamiltonian(boundaries)
        energy = ham.expectation(state)

        return energy

    def get_magnetization(self, state, boundaries):
        q1 = self.get_q1(boundaries)
        magnetization = q1.expectation(state)

        return magnetization

    def get_correlation(self, state, boundaries):
        q2 = self.get_q2(boundaries)
        correlation = q2.expectation(state)

        return correlation
    
    def get_ej_expectation(self, j, state, boundaries):
        ej = self.get_ej(j, boundaries)
        ej_expectation = ej.expectation(state)

        return ej_expectation
    
    def get_nonlocal_pauli_expectation(self, num, state, boundaries):
        obs = self.get_nonlocal_pauli(num, boundaries)
        obs_expectation = obs.expectation(state)

        return obs_expectation

    def circ_to_qiskit(self, circ, measure_all=False):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import XGate, SwapGate, CXGate, UnitaryGate, CZGate, RZGate, SXGate

        backend = construct_backend('numpy')
        gate_list = circ.queue
        if measure_all:
            circ_qiskit = QuantumCircuit(circ.nqubits, circ.nqubits)
        else:
            circ_qiskit = QuantumCircuit(circ.nqubits, self.N)
        for g in gate_list:
            control_qubits = g.control_qubits[::-1]
            target_qubits = g.target_qubits[::-1]
            if isinstance(g, gates.X) or isinstance(g, gates.SWAP) or isinstance(g, gates.TOFFOLI) or isinstance(g, gates.RZ) or isinstance(g, gates.SX):
                if isinstance(g, gates.X):
                    g1 = XGate()
                elif isinstance(g, gates.SWAP):
                    g1 = SwapGate()
                elif isinstance(g, gates.TOFFOLI):
                    g1 = XGate()
                elif isinstance(g, gates.RZ):
                    g1 = RZGate(g.parameters[0])
                elif isinstance(g, gates.SX):
                    g1 = SXGate()
                if len(control_qubits) == 0:
                    circ_qiskit.append(g1, target_qubits)
                else:
                    circ_qiskit.append(g1.control(
                        len(control_qubits)), control_qubits+target_qubits)
            elif isinstance(g, gates.CNOT):
                circ_qiskit.append(CXGate(), control_qubits+target_qubits)
            elif isinstance(g, gates.CZ):
                circ_qiskit.append(CZGate(), control_qubits+target_qubits)
            elif isinstance(g, gates.Unitary) or isinstance(g, gates.GeneralizedfSim):
                matrix = g.matrix(backend)
                if len(control_qubits) == 0:
                    circ_qiskit.append(UnitaryGate(matrix), target_qubits)
                else:
                    circ_qiskit.append(UnitaryGate(matrix).control(
                        len(control_qubits)), control_qubits+target_qubits)
            else:
                raise ValueError(f"Unsupported gate type: {type(g)}")
                    
        return circ_qiskit
    
    def circ_to_quantinuum(self, circ, measure_all=False):
        from pytket.circuit import Circuit, OpType, QControlBox, Unitary2qBox, Unitary1qBox, Op

        backend = construct_backend('numpy')
        gate_list = circ.queue
        if measure_all:
            circ_quantinuum = Circuit(circ.nqubits, circ.nqubits)
        else:
            circ_quantinuum = Circuit(circ.nqubits, self.N)
        for g in gate_list:
            control_qubits = g.control_qubits
            target_qubits = g.target_qubits
            if isinstance(g, (gates.X, gates.SWAP, gates.TOFFOLI, gates.SX, gates.RZ)):
                if isinstance(g, gates.X):
                    g1 = OpType.X
                elif isinstance(g, gates.SWAP):
                    g1 = OpType.SWAP
                elif isinstance(g, gates.TOFFOLI):
                    g1 = OpType.X
                elif isinstance(g, gates.SX):
                    g1 = OpType.SX
                elif isinstance(g, gates.RZ):
                    g1 = OpType.Rz
                if len(control_qubits) == 0:
                    if isinstance(g, gates.RZ):
                        circ_quantinuum.add_gate(g1, g.parameters[0]/np.pi, target_qubits)
                    else:
                        circ_quantinuum.add_gate(g1, target_qubits)
                else:
                    if isinstance(g, gates.RZ):
                        circ_quantinuum.add_gate(QControlBox(Op.create(g1),n_controls=len(control_qubits)), g.parameters[0]/np.pi, control_qubits+target_qubits)
                    else:
                        circ_quantinuum.add_gate(QControlBox(Op.create(g1),n_controls=len(control_qubits)), control_qubits+target_qubits)
            elif isinstance(g, gates.CNOT):
                circ_quantinuum.add_gate(OpType.CX, control_qubits+target_qubits)
            elif isinstance(g, gates.CZ):
                circ_quantinuum.add_gate(OpType.CZ, control_qubits+target_qubits)
            elif isinstance(g, gates.Unitary) or isinstance(g, gates.GeneralizedfSim):
                matrix = g.matrix(backend)
                if len(target_qubits) == 1: 
                    g1 = Unitary1qBox
                else:
                    g1 = Unitary2qBox
                if len(control_qubits) == 0:
                    circ_quantinuum.add_gate(g1(matrix), target_qubits)
                else:
                    circ_quantinuum.add_gate(QControlBox(g1(matrix),n_controls=len(control_qubits)), control_qubits+target_qubits)
                    
        return circ_quantinuum

    def sample_circuit_qiskit(self, device_backend, transpiler, shots, coupling_map, basis_gates, layout):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from qiskit import transpile

        if self.M == 1:
            keep = [0]+[2*(j+1) for j in range(self.N)] + \
                [self.circ_full.nqubits-4]
        else:
            keep = [0]+[2*(j+1) for j in range(self.N)] + \
                [self.circ_full.nqubits-5-self.M]
        circ = self.circ_full
        circ_qiskit = self.circ_to_qiskit(circ)

        circ_qiskit_z = circ_qiskit.copy()
        circ_qiskit_z.measure(keep, list(range(len(keep)))[::-1])

        circ_qiskit_x = circ_qiskit.copy()
        for q in keep:
            circ_qiskit_x.h(q)
        circ_qiskit_x.measure(keep, list(range(len(keep)))[::-1])

        circ_qiskit_y = circ_qiskit.copy()
        for q in keep:
            circ_qiskit_y.sdg(q)
            circ_qiskit_y.h(q)
        circ_qiskit_y.measure(keep, list(range(len(keep)))[::-1])

        if device_backend is not None:
            noise = NoiseModel().from_backend(device_backend)
            if transpiler:
                circ_qiskit_z = transpile(circ_qiskit_z, basis_gates=basis_gates,
                                          coupling_map=coupling_map,  initial_layout=layout, optimization_level=3)
                circ_qiskit_x = transpile(circ_qiskit_x, basis_gates=basis_gates,
                                          coupling_map=coupling_map,  initial_layout=layout, optimization_level=3)
                circ_qiskit_y = transpile(circ_qiskit_y, basis_gates=basis_gates,
                                          coupling_map=coupling_map,  initial_layout=layout, optimization_level=3)
        else:
            noise = None
        sim = AerSimulator(method='statevector', noise_model=noise)

        result_z = sim.run(circ_qiskit_z, shots=shots).result()
        counts_z = result_z.get_counts(0)

        result_x = sim.run(circ_qiskit_x, shots=shots).result()
        counts_x = result_x.get_counts(0)

        result_y = sim.run(circ_qiskit_y, shots=shots).result()
        counts_y = result_y.get_counts(0)

        return counts_x, counts_y, counts_z

    def sample_circuit(self, nshots, noise_model, layout, boundaries=True, backend=None):
        if boundaries:
            if self.M == 1:
                keep = [self.circ_full.nqubits-3]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                    self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-4]
            else:
                keep = [self.circ_full.nqubits-4]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                    self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-5-self.M]
        else:
            if self.M == 1:
                keep = [2*j+1 for j in range(self.N-self.D)] + \
                    [2*(self.N-self.D)+i for i in range(self.D)]
            else:
                keep = [2*j+1 for j in range(self.N-self.D)] + \
                    [2*(self.N-self.D)+i for i in range(self.D)]

        if layout is not None:
            keep = [layout[k] for k in keep]

        circ = self.circ_full
        if noise_model is not None:
            circ = noise_model.apply(circ)

        circ_z = circ.copy()
        circ_z.add(gates.M(*keep))

        circ_x = circ.copy()
        for q in keep:
            circ_x.add(gates.H(q))
        circ_x.add(gates.M(*keep))

        circ_y = circ.copy()
        for q in keep:
            circ_y.add(gates.SDG(q))
            circ_y.add(gates.H(q))
        circ_y.add(gates.M(*keep))

        backend = _check_backend(backend)

        result_z = backend.execute_circuit(circ_z, nshots=nshots)
        counts_z = result_z.frequencies()

        result_x = backend.execute_circuit(circ_x, nshots=nshots)
        counts_x = result_x.frequencies()

        result_y = backend.execute_circuit(circ_y, nshots=nshots)
        counts_y = result_y.frequencies()

        return counts_x, counts_y, counts_z
    
    def sample_circuit(self, nshots, noise_model, layout, boundaries=True, backend=None, error_detection = False):
        # error detection only works for D=0
        if self.D != 0:
            if boundaries:
                if (self.N == 5 and self.M == 1 and self.D == 2) or (self.N == 6 and self.M == 1 and self.D == 2):
                    keep = [self.circ_full.nqubits-1]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-2]
                else:
                    keep = [self.circ_full.nqubits-2-(int(self.D/2) + 2 + int(self.D/2) + 1)]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-1-(int(self.D/2) + 2 + int(self.D/2) + 1)]
            else:
                if self.M == 1:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
                else:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
        else:
            if boundaries:
                raise ValueError(
                    'Boundaries not implemented for D=0')
            else:
                keep = list(range(self.N))
                aux = list(range(self.N,self.N+self.M+1))
                if error_detection:
                    keep_aux = keep + aux
                else:
                    keep_aux = keep


        if layout is not None:
            keep = [layout[k] for k in keep]
            aux = [layout[k] for k in aux]
            if self.D == 0 and error_detection:
                keep_aux = keep + aux
            else:
                keep_aux = keep

        circ = self.circ_full

        circ_z = circ.copy()
        #circ_z.add(gates.M(*keep_aux))
        circ_z.add([gates.M(q) for q in keep_aux])

        circ_x = circ.copy()
        for q in keep:
            circ_x.add(gates.H(q))
        circ_x.add([gates.M(q) for q in keep_aux])

        circ_y = circ.copy()
        for q in keep:
            circ_y.add(gates.SDG(q))
            circ_y.add(gates.H(q))
        circ_y.add([gates.M(q) for q in keep_aux])

        if noise_model is not None:
            circ_z = noise_model.apply(circ_z)
            circ_z.density_matrix = True
            circ_x = noise_model.apply(circ_x)
            circ_x.density_matrix = True
            circ_y = noise_model.apply(circ_y)
            circ_y.density_matrix = True

        backend = _check_backend(backend)

        result_z = backend.execute_circuit(circ_z, nshots=nshots)
        counts_z = result_z.frequencies()

        result_x = backend.execute_circuit(circ_x, nshots=nshots)
        counts_x = result_x.frequencies()

        result_y = backend.execute_circuit(circ_y, nshots=nshots)
        counts_y = result_y.frequencies()

        if self.D == 0 and error_detection:
            counts_x = self.get_nsites_counts(counts_x, error_detection)
            counts_y = self.get_nsites_counts(counts_y, error_detection)
            counts_z = self.get_nsites_counts(counts_z, error_detection)

        # counts zxxz
        # counts zyyz
            
        num_iter = 3 #int((self.N-self.N%3)/3) - 1

        output_list = []
        new_indices_list = []
        for i in range(num_iter):
            if i == 0:
                output = self.create_repeated_string()
            else:
                output = 'x'+output[:self.N-1]
            indices = [i for i, char in enumerate(output) if char == 'z']
            new_indices = []
            for j, index in enumerate(indices):
                if index != 2 and index != self.N - 3:
                    if output[indices[j]:indices[j]+4] == 'zxxz':
                        new_indices.append(index)
                elif index == 2:
                    if output[0:indices[j]+1] == 'xxz':
                        new_indices.append('left_b')
                        new_indices.append(index)
                elif index == self.N - 3:
                    if output[indices[j]:] == 'zxx':
                        #new_indices.append(index)
                        new_indices.append('right_b')

            output_list.append(output)
            new_indices_list.append(new_indices)

        circ_x_list = []
        circ_y_list = []

        for string in output_list:
            circ_zxxz = circ.copy()
            circ_zyyz = circ.copy()
            for j, q in enumerate(keep):
                if string[j] != 'z':
                    circ_zxxz.add(gates.H(q)) #measure x in zxxz

                    circ_zyyz.add(gates.SDG(q)) #measure y in zyyz instead of x
                    circ_zyyz.add(gates.H(q))

            #circ_zxxz.add(gates.M(*keep_aux))
            circ_zxxz.add([gates.M(q) for q in keep_aux])
            circ_zyyz.add([gates.M(q) for q in keep_aux])

            if noise_model is not None:
                circ_zxxz = noise_model.apply(circ_zxxz)
                circ_zxxz.density_matrix = True
                circ_zyyz = noise_model.apply(circ_zyyz)
                circ_zyyz.density_matrix = True

            circ_x_list.append(circ_zxxz)
            circ_y_list.append(circ_zyyz)

        #execute circs zxxz and zyyz
        counts_zxxz_list = []
        counts_zyyz_list = []
        for circ_zxxz, circ_zyyz in zip(circ_x_list, circ_y_list):

            result_zxxz = backend.execute_circuit(circ_zxxz, nshots=nshots)
            counts_zxxz = result_zxxz.frequencies()
    
            result_zyyz = backend.execute_circuit(circ_zyyz, nshots=nshots)
            counts_zyyz = result_zyyz.frequencies()

            if self.D == 0 and error_detection:
                counts_zxxz = self.get_nsites_counts(counts_zxxz, error_detection)
                counts_zyyz = self.get_nsites_counts(counts_zyyz, error_detection)

            counts_zxxz_list.append(counts_zxxz)
            counts_zyyz_list.append(counts_zyyz)

        return counts_x, counts_y, counts_z, [new_indices_list, counts_zxxz_list, counts_zyyz_list] 
    
    def count_transitions(self, binary_string):
        count = 0
        if binary_string[0] == '1':
            count += 1
        if binary_string[-1] == '1':
            count += 1
        for i in range(1, len(binary_string)):
            if (binary_string[i-1] == '0' and binary_string[i] == '1') or \
            (binary_string[i-1] == '1' and binary_string[i] == '0'):
                count += 1
        return count
    
    # def count_transitions(self, binary_string):
    #     count_odd = 0
    #     count_even = 0
    #     for k in range(len(binary_string)-1):
    #         if binary_string[k] != binary_string[k+1]:
    #             if k % 2 == 0:
    #                 count_even += 1
    #             else:
    #                 count_odd += 1

    #     if binary_string[0] == '1':
    #         count_odd += 1
    #     if len(binary_string) % 2 == 0:
    #         if binary_string[-1] == '1':
    #             count_odd += 1
    #     else:
    #         if binary_string[-1] == '1':
    #             count_even += 1

    #     return count_even, count_odd

    def get_nsites_counts(self, counts, postselect=True, postselect_aux=False, mode=None):
        # only for D=0
        from collections import Counter
        new_counts = Counter()
        # for key, value in counts.items():
        #     if postselect and mode=='not_z':
        #         if key[self.N:self.N+self.M+1] == '0'*self.M+'1':
        #             new_counts[key[0:self.N]] = value
        #     else:
        #         new_counts[key[0:self.N]] = new_counts.get(key[0:self.N], 0) + value

        # if postselect and mode=='z':
        #     new_counts2 = Counter()
        #     for key, value in new_counts.items():
        #         if key.count('1') == self.M:
        #             new_counts2[key] = value
        #     new_counts = new_counts2

        if postselect:
            new_counts2 = Counter()
            for key, value in new_counts.items():
                if key.count('1') == self.M and key[self.N:self.N+self.M+1] == '0'*self.M+'1':
                    new_counts2[key[0:self.N]] = value
            new_counts = new_counts2

        if postselect:
            print('survival counts post q1',np.sum(list(new_counts.values()))/np.sum(list(counts.values())))

        if postselect_aux:
            new_counts2 = Counter()
            for key, value in new_counts.items():
                if mode=='not_z' and self.count_transitions(key) <= int(2*self.M):
                    new_counts2[key] = value
                elif mode=='z' and self.count_transitions(key) == int(2*self.M):
                    new_counts2[key] = value

            new_counts = new_counts2        

            print('survival counts post q1 and q2',np.sum(list(new_counts.values()))/np.sum(list(counts.values())))

        return new_counts
    

    def get_nsites_counts(self, counts, postselect=True, postselect_aux=False, mode=None): #new mode
        # only for D=0
        from collections import Counter
        new_counts = Counter()
        for key, value in counts.items():
            if postselect: #and mode=='not_z':
                if key[0:self.N].count('1') == self.M and key[self.N:self.N+self.M+1] == '0'*self.M+'1':
                    new_counts[key[0:self.N]] = value
            else:
                new_counts[key[0:self.N]] = new_counts.get(key[0:self.N], 0) + value


        if postselect_aux:
            new_counts2 = Counter()
            for key, value in new_counts.items():
                if mode=='not_z' and self.count_transitions(key) <= int(2*self.M):
                    new_counts2[key] = value
                elif mode=='z' and self.count_transitions(key) == int(2*self.M):
                    new_counts2[key] = value

            new_counts = new_counts2        

        survival_ratio = np.sum(list(new_counts.values()))/np.sum(list(counts.values()))
        if postselect and postselect_aux:
            print('survival counts post q1 and q2', survival_ratio)
        elif postselect:
            print('survival counts post q1', survival_ratio)
        elif postselect_aux:
            print('survival counts post q2', survival_ratio)

        return new_counts
    
    def create_repeated_string(self):
        pattern = "zxx"
        result = (pattern * ((self.N) // len(pattern) + 1))[:self.N]
        return result

    def sample_circuit_quantinuum(self, device, nshots, layout, boundaries=False,  measure_all = False, compile=True, circ_to_quantinuum=True, optimization_level=3): #it works without boundaries
        
        if self.D != 0:
            if boundaries:
                if (self.N == 5 and self.M == 1 and self.D == 2) or (self.N == 6 and self.M == 1 and self.D == 2):
                    keep = [self.circ_full.nqubits-1]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-2]
                else:
                    keep = [self.circ_full.nqubits-2-(int(self.D/2) + 2 + int(self.D/2) + 1)]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-1-(int(self.D/2) + 2 + int(self.D/2) + 1)]
            else:
                if self.M == 1:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
                else:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
        else:
            if boundaries:
                raise ValueError(
                    'Boundaries not implemented for D=0')
            else:
                keep = list(range(self.N))
                aux = list(range(self.N,self.N+self.M+1))
                if measure_all:
                    keep_aux = keep + aux
                else:
                    keep_aux = keep

        if layout is not None:
            keep = [layout[k] for k in keep]
            aux = [layout[k] for k in aux]
            if self.D == 0 and measure_all:
                keep_aux = keep + aux
            else:
                keep_aux = keep

        circ = self.circ_full
        if circ_to_quantinuum:
            circ = self.circ_to_quantinuum(circ, measure_all=measure_all)


        # from pytket.extensions.qiskit import AerStateBackend
        # aer_state_b = AerStateBackend()
        # circ_quantinuum = aer_state_b.get_compiled_circuit(circ_quantinuum)

        # state_handle = aer_state_b.process_circuit(circ_quantinuum)
        # statevector = aer_state_b.get_result(state_handle).get_state()
        # circ.density_matrix = False
        # from qibo.backends import construct_backend
        # backend = construct_backend("qibojit",platform='numba')
        # from qibo.quantum_info import fidelity
        # print(fidelity(backend.execute_circuit(circ).state(), statevector, backend=backend))

        circ_z = circ.copy()
        circ_z.measure_all
        for j, q in enumerate(keep_aux):
            circ_z.Measure(q, j)

        circ_x = circ.copy()
        for q in keep:
            if compile:
                circ_x.H(q)
            else:
                circ_x.PhasedX(0.5, 1.5, q)
                circ_x.Rz(1, q)
        for j, q in enumerate(keep_aux):
            circ_x.Measure(q, j)

        circ_y = circ.copy()
        for q in keep:
            if compile:
                circ_y.Sdg(q)
                circ_y.H(q)
            else:
                circ_y.PhasedX(0.5, 0, q)
                circ_y.Rz(0.5, q)
        for j, q in enumerate(keep_aux):
            circ_y.Measure(q, j)

                
        # circ_z = device_backend.get_compiled_circuit(circ_z, optimisation_level=2)
        # circ_x = device_backend.get_compiled_circuit(circ_x, optimisation_level=2)
        # circ_y = device_backend.get_compiled_circuit(circ_y, optimisation_level=2)
        # print('depth quantinuum', circ_z.depth())
        # print('1q gates quantinuum', circ_z.n_1qb_gates())
        # print('2q gates quantinuum', circ_z.n_2qb_gates())

            
        num_iter = 3 #int((self.N-self.N%3)/3) - 1

        output_list = []
        new_indices_list = []
        for i in range(num_iter):
            if i == 0:
                output = self.create_repeated_string()
            else:
                output = 'x'+output[:self.N-1]
            indices = [i for i, char in enumerate(output) if char == 'z']
            new_indices = []
            for j, index in enumerate(indices):
                if index != 2 and index != self.N - 3:
                    if output[indices[j]:indices[j]+4] == 'zxxz':
                        new_indices.append(index)
                elif index == 2:
                    if output[0:indices[j]+1] == 'xxz':
                        new_indices.append('left_b')
                        new_indices.append(index)
                elif index == self.N - 3:
                    if output[indices[j]:] == 'zxx':
                        #new_indices.append(index)
                        new_indices.append('right_b')

            output_list.append(output)
            new_indices_list.append(new_indices)

        circ_x_list = []
        circ_y_list = []

        for string in output_list:
            circ_zxxz = circ.copy()
            circ_zyyz = circ.copy()
            for j, q in enumerate(keep):
                if string[j] != 'z':
                    if compile:
                        circ_zxxz.H(q) #measure x in zxxz
                    else:
                        circ_zxxz.PhasedX(0.5, 1.5, q)
                        circ_zxxz.Rz(1, q)

                    if compile:
                        circ_zyyz.Sdg(q) #measure y in zyyz instead of x
                        circ_zyyz.H(q)
                    else:
                        circ_zyyz.PhasedX(0.5, 0, q)
                        circ_zyyz.Rz(0.5, q)

            for j, q in enumerate(keep_aux):
                    circ_zxxz.Measure(q, j)
                    circ_zyyz.Measure(q, j)
            circ_x_list.append(circ_zxxz)
            circ_y_list.append(circ_zyyz)


        # Execute all circuits in one batch

        circs = [circ_z, circ_x, circ_y] + circ_x_list + circ_y_list

        #optimization_level = 3
        name_project = "XXZ_folded"
        results, compiled_circuits = compile_quantinuum(circs, name_project, optimization_level, nshots, device, compile, counts=False)

        counts_z = results[0].get_counts()
        counts_x = results[1].get_counts()
        counts_y = results[2].get_counts()

        counts_z = counts_to_qibo(counts_z)
        counts_x = counts_to_qibo(counts_x)
        counts_y = counts_to_qibo(counts_y)
            
        counts_zxxz_list = []
        counts_zyyz_list = []
        for i in range(num_iter):
            counts_zxxz = results[i+3].get_counts()
            counts_zyyz = results[i+num_iter+3].get_counts()
            counts_zxxz = counts_to_qibo(counts_zxxz)
            counts_zyyz = counts_to_qibo(counts_zyyz)
            counts_zxxz_list.append(counts_zxxz)
            counts_zyyz_list.append(counts_zyyz)

        return counts_x, counts_y, counts_z, [new_indices_list, counts_zxxz_list, counts_zyyz_list], compiled_circuits 
    

    def sample_circuit_quantinuum_postq2(self, device, nshots, layout, boundaries=False,  measure_all = False, compile=True, circ_to_quantinuum=True, optimization_level=3): #it works without boundaries
        
        if self.D != 0:
            if boundaries:
                if (self.N == 5 and self.M == 1 and self.D == 2) or (self.N == 6 and self.M == 1 and self.D == 2):
                    keep = [self.circ_full.nqubits-1]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-2]
                else:
                    keep = [self.circ_full.nqubits-2-(int(self.D/2) + 2 + int(self.D/2) + 1)]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-1-(int(self.D/2) + 2 + int(self.D/2) + 1)]
            else:
                if self.M == 1:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
                else:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
        else:
            if boundaries:
                raise ValueError(
                    'Boundaries not implemented for D=0')
            else:
                keep = list(range(self.N))
                aux = list(range(self.N,self.N+self.M+1))
                if measure_all:
                    keep_aux = keep + aux
                else:
                    keep_aux = keep

        if layout is not None:
            keep = [layout[k] for k in keep]
            aux = [layout[k] for k in aux]
            if self.D == 0 and measure_all:
                keep_aux = keep + aux
            else:
                keep_aux = keep

        circ = self.circ_full
        if circ_to_quantinuum:
            circ = self.circ_to_quantinuum(circ, measure_all=measure_all)


        # from pytket.extensions.qiskit import AerStateBackend
        # aer_state_b = AerStateBackend()
        # circ_quantinuum = aer_state_b.get_compiled_circuit(circ_quantinuum)

        # state_handle = aer_state_b.process_circuit(circ_quantinuum)
        # statevector = aer_state_b.get_result(state_handle).get_state()
        # circ.density_matrix = False
        # from qibo.backends import construct_backend
        # backend = construct_backend("qibojit",platform='numba')
        # from qibo.quantum_info import fidelity
        # print(fidelity(backend.execute_circuit(circ).state(), statevector, backend=backend))

        circ_z = circ.copy()
        circ_z.measure_all
        for j, q in enumerate(keep_aux):
            circ_z.Measure(q, j)

        from pytket.circuit import Unitary2qBox

        rotation_gate = gates.GIVENS(0,1,np.pi/4)
        matrix = rotation_gate.matrix(self.backend)
        g1 = Unitary2qBox
        circ_x_y_even = circ.copy()
        circ_x_y_odd = circ.copy()
        for index, q in enumerate(keep[0:-1]):
            if index % 2 == 0:
                #circ_x_y_even.add(rotation_gate.on_qubits({0:keep[index],1:keep[index+1]}))
                circ_x_y_even.add_gate(g1(matrix), [keep[index], keep[index+1]])
            else:
                #circ_x_y_odd.add(rotation_gate.on_qubits({0:keep[index],1:keep[index+1]}))
                circ_x_y_odd.add_gate(g1(matrix), [keep[index], keep[index+1]])
        for j, q in enumerate(keep_aux):
            circ_x_y_even.Measure(q, j)
            circ_x_y_odd.Measure(q, j)

        # circ_z = device_backend.get_compiled_circuit(circ_z, optimisation_level=2)
        # circ_x = device_backend.get_compiled_circuit(circ_x, optimisation_level=2)
        # circ_y = device_backend.get_compiled_circuit(circ_y, optimisation_level=2)
        # print('depth quantinuum', circ_z.depth())
        # print('1q gates quantinuum', circ_z.n_1qb_gates())
        # print('2q gates quantinuum', circ_z.n_2qb_gates())

            
        num_iter = 3 #int((self.N-self.N%3)/3) - 1

        output_list = []
        new_indices_list = []
        for i in range(num_iter):
            if i == 0:
                output = self.create_repeated_string()
            else:
                output = 'x'+output[:self.N-1]
            indices = [i for i, char in enumerate(output) if char == 'z']
            new_indices = []
            for j, index in enumerate(indices):
                if index != 2 and index != self.N - 3:
                    if output[indices[j]:indices[j]+4] == 'zxxz':
                        new_indices.append(index)
                elif index == 2:
                    if output[0:indices[j]+1] == 'xxz':
                        new_indices.append('left_b')
                        new_indices.append(index)
                elif index == self.N - 3:
                    if output[indices[j]:] == 'zxx':
                        #new_indices.append(index)
                        new_indices.append('right_b')

            output_list.append(output)
            new_indices_list.append(new_indices)

        circ_xy_list = []
        for string in output_list:
            circ_zxxz_zyyz = circ.copy()
            for j, q in enumerate(keep[0:-1]):
                if string[j] == 'x' and string[j+1] == 'x':
                    #circ_zxxz_zyyz.add(rotation_gate.on_qubits({0:keep[j],1:keep[j+1]}))    
                    circ_zxxz_zyyz.add_gate(g1(matrix), [keep[j], keep[j+1]])
            for j, q in enumerate(keep_aux):
                    circ_zxxz_zyyz.Measure(q, j)

            circ_xy_list.append(circ_zxxz_zyyz)


        # Execute all circuits in one batch

        circs = [circ_z, circ_x_y_even, circ_x_y_odd] + circ_xy_list 

        #optimization_level = 3
        name_project = "XXZ_folded"
        results, compiled_circuits = compile_quantinuum(circs, name_project, optimization_level, nshots, device, compile, counts=False)

        counts_z = results[0].get_counts()
        counts_x_y_even = results[1].get_counts()
        counts_x_y_odd = results[2].get_counts()

        counts_z = counts_to_qibo(counts_z)
        counts_x_y_even = counts_to_qibo(counts_x_y_even)
        counts_x_y_odd = counts_to_qibo(counts_x_y_odd)
            
        counts_zxxz_zyyz_list = []
        for i in range(num_iter):
            counts_zxxz_zyyz = results[i+3].get_counts()
            counts_zxxz_zyyz = counts_to_qibo(counts_zxxz_zyyz)
            counts_zxxz_zyyz_list.append(counts_zxxz_zyyz )

        return counts_x_y_even, counts_x_y_odd, counts_z, [new_indices_list, counts_zxxz_zyyz_list], compiled_circuits 
    
        
    def sample_circuit_ionq(self, device, nshots, layout, boundaries=False,  measure_all = False, compile=False, circ_to_ionq=True, optimization_level=3): #it works without boundaries
            
            if self.D != 0:
                if boundaries:
                    if (self.N == 5 and self.M == 1 and self.D == 2) or (self.N == 6 and self.M == 1 and self.D == 2):
                        keep = [self.circ_full.nqubits-1]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                            self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-2]
                    else:
                        keep = [self.circ_full.nqubits-2-(int(self.D/2) + 2 + int(self.D/2) + 1)]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                            self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-1-(int(self.D/2) + 2 + int(self.D/2) + 1)]
                else:
                    if self.M == 1:
                        keep = [2*j+1 for j in range(self.N-self.D)] + \
                            [2*(self.N-self.D)+i for i in range(self.D)]
                    else:
                        keep = [2*j+1 for j in range(self.N-self.D)] + \
                            [2*(self.N-self.D)+i for i in range(self.D)]
            else:
                if boundaries:
                    raise ValueError(
                        'Boundaries not implemented for D=0')
                else:
                    keep = list(range(self.N))
                    aux = list(range(self.N,self.N+self.M+1))
                    if measure_all:
                        keep_aux = keep + aux
                    else:
                        keep_aux = keep

            if layout is not None:
                keep = [layout[k] for k in keep]
                aux = [layout[k] for k in aux]
                if self.D == 0 and measure_all:
                    keep_aux = keep + aux
                else:
                    keep_aux = keep

            circ = self.circ_full
            if circ_to_ionq:
                circ = self.circ_to_qiskit(circ, measure_all=measure_all)


            # from pytket.extensions.qiskit import AerStateBackend
            # aer_state_b = AerStateBackend()
            # circ_quantinuum = aer_state_b.get_compiled_circuit(circ_quantinuum)

            # state_handle = aer_state_b.process_circuit(circ_quantinuum)
            # statevector = aer_state_b.get_result(state_handle).get_state()
            # circ.density_matrix = False
            # from qibo.backends import construct_backend
            # backend = construct_backend("qibojit",platform='numba')
            # from qibo.quantum_info import fidelity
            # print(fidelity(backend.execute_circuit(circ).state(), statevector, backend=backend))
            from qiskit_ionq import GPIGate, GPI2Gate, MSGate

            circ_z = circ.copy()
            for j, q in enumerate(keep_aux):
                circ_z.measure(q, j)

            circ_x = circ.copy()
            for q in keep:
                circ_x.h(q)
                if compile:
                    circ_x.h(q)
                # else:
                #     circ_x.append(GPI2Gate(0),[q])
                #     circ_x.append(GPIGate(-0.125),[q])
                #     circ_x.append(GPI2Gate(0.5),[q])
            for j, q in enumerate(keep_aux):
                circ_x.measure(q, j)

            circ_y = circ.copy()
            for q in keep:
                circ_y.sdg(q)
                circ_y.h(q)
                # if compile:
                #     circ_y.sdg(q)
                #     circ_y.h(q)
                # else:
                #     circ_y.append(GPI2Gate(0.75),[q])
                #     circ_y.append(GPIGate(0.125),[q])
                #     circ_y.append(GPI2Gate(0.5),[q])
                #     circ_y.append(GPI2Gate(0),[q])
                #     circ_y.append(GPIGate(-0.125),[q])
                #     circ_y.append(GPI2Gate(0.5),[q])
            for j, q in enumerate(keep_aux):
                circ_y.measure(q, j)

                    
            # circ_z = device_backend.get_compiled_circuit(circ_z, optimisation_level=2)
            # circ_x = device_backend.get_compiled_circuit(circ_x, optimisation_level=2)
            # circ_y = device_backend.get_compiled_circuit(circ_y, optimisation_level=2)
            # print('depth quantinuum', circ_z.depth())
            # print('1q gates quantinuum', circ_z.n_1qb_gates())
            # print('2q gates quantinuum', circ_z.n_2qb_gates())

                
            num_iter = 3 #int((self.N-self.N%3)/3) - 1

            output_list = []
            new_indices_list = []
            for i in range(num_iter):
                if i == 0:
                    output = self.create_repeated_string()
                else:
                    output = 'x'+output[:self.N-1]
                indices = [i for i, char in enumerate(output) if char == 'z']
                new_indices = []
                for j, index in enumerate(indices):
                    if index != 2 and index != self.N - 3:
                        if output[indices[j]:indices[j]+4] == 'zxxz':
                            new_indices.append(index)
                    elif index == 2:
                        if output[0:indices[j]+1] == 'xxz':
                            new_indices.append('left_b')
                            new_indices.append(index)
                    elif index == self.N - 3:
                        if output[indices[j]:] == 'zxx':
                            #new_indices.append(index)
                            new_indices.append('right_b')

                output_list.append(output)
                new_indices_list.append(new_indices)

            circ_x_list = []
            circ_y_list = []

            for string in output_list:
                circ_zxxz = circ.copy()
                circ_zyyz = circ.copy()
                for j, q in enumerate(keep):
                    if string[j] != 'z':
                        if compile:
                            circ_zxxz.h(q) #measure x in zxxz
                        else:
                            circ_zxxz.append(GPI2Gate(0),[q])
                            circ_zxxz.append(GPIGate(-0.125),[q])
                            circ_zxxz.append(GPI2Gate(0.5),[q])

                        if compile:
                            circ_zyyz.sdg(q) #measure y in zyyz instead of x
                            circ_zyyz.h(q)
                        else:
                            circ_zyyz.PhasedX(0.5, 0, q)
                            circ_zyyz.append(GPI2Gate(0.75),[q])
                            circ_zyyz.append(GPIGate(0.125),[q])
                            circ_zyyz.append(GPI2Gate(0.5),[q])
                            circ_zyyz.append(GPI2Gate(0),[q])
                            circ_zyyz.append(GPIGate(-0.125),[q])
                            circ_zyyz.append(GPI2Gate(0.5),[q])

                for j, q in enumerate(keep_aux):
                        circ_zxxz.measure(q, j)
                        circ_zyyz.measure(q, j)
                circ_x_list.append(circ_zxxz)
                circ_y_list.append(circ_zyyz)


            # Execute all circuits in one batch

            circs = [circ_z, circ_x, circ_y] + circ_x_list + circ_y_list

            #optimization_level = 3
            name_project = "XXZ_folded"
            results, compiled_circuits = compile_ionq(circs, optimization_level, nshots, device, compile, counts=False)

            counts_z = results.get_counts(circs[0])
            counts_x = results.get_counts(circs[1])
            counts_y = results.get_counts(circs[2])

            # counts_z = counts_to_qibo(counts_z)
            # counts_x = counts_to_qibo(counts_x)
            # counts_y = counts_to_qibo(counts_y)
                
            counts_zxxz_list = []
            counts_zyyz_list = []
            for i in range(num_iter):
                counts_zxxz = results.get_counts(circs[i+3])
                counts_zyyz = results.get_counts(circs[i+num_iter+3])
                # counts_zxxz = counts_to_qibo(counts_zxxz)
                # counts_zyyz = counts_to_qibo(counts_zyyz)
                counts_zxxz_list.append(counts_zxxz)
                counts_zyyz_list.append(counts_zyyz)

            return counts_x, counts_y, counts_z, [new_indices_list, counts_zxxz_list, counts_zyyz_list], compiled_circuits 


    def sample_circuit_ionq_postq2(self, device, nshots, layout, boundaries=False,  measure_all = False, compile=True, circ_to_ionq=True, optimization_level=3): #it works without boundaries
        
        if self.D != 0:
            if boundaries:
                if (self.N == 5 and self.M == 1 and self.D == 2) or (self.N == 6 and self.M == 1 and self.D == 2):
                    keep = [self.circ_full.nqubits-1]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-2]
                else:
                    keep = [self.circ_full.nqubits-2-(int(self.D/2) + 2 + int(self.D/2) + 1)]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                        self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-1-(int(self.D/2) + 2 + int(self.D/2) + 1)]
            else:
                if self.M == 1:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
                else:
                    keep = [2*j+1 for j in range(self.N-self.D)] + \
                        [2*(self.N-self.D)+i for i in range(self.D)]
        else:
            if boundaries:
                raise ValueError(
                    'Boundaries not implemented for D=0')
            else:
                keep = list(range(self.N))
                aux = list(range(self.N,self.N+self.M+1))
                if measure_all:
                    keep_aux = keep + aux
                else:
                    keep_aux = keep

        if layout is not None:
            keep = [layout[k] for k in keep]
            aux = [layout[k] for k in aux]
            if self.D == 0 and measure_all:
                keep_aux = keep + aux
            else:
                keep_aux = keep

        circ = self.circ_full
        if circ_to_ionq:
            circ = self.circ_to_qiskit(circ, measure_all=measure_all)


        # from pytket.extensions.qiskit import AerStateBackend
        # aer_state_b = AerStateBackend()
        # circ_quantinuum = aer_state_b.get_compiled_circuit(circ_quantinuum)

        # state_handle = aer_state_b.process_circuit(circ_quantinuum)
        # statevector = aer_state_b.get_result(state_handle).get_state()
        # circ.density_matrix = False
        # from qibo.backends import construct_backend
        # backend = construct_backend("qibojit",platform='numba')
        # from qibo.quantum_info import fidelity
        # print(fidelity(backend.execute_circuit(circ).state(), statevector, backend=backend))

        circ_z = circ.copy()
        #circ_z.measure_all()
        # for j, q in enumerate(keep_aux):
        #     circ_z.measure(q, j)

        circ_z.measure(keep_aux, list(range(len(keep_aux)))[::-1])

        from qiskit.circuit.library import UnitaryGate, RXXGate

        rotation_gate = gates.GIVENS(0,1,np.pi/4)
        matrix = rotation_gate.matrix(self.backend)
        g1 = UnitaryGate(matrix)



        #g1 = RXXGate(np.pi/4)
        circ_x_y_even = circ.copy()
        circ_x_y_odd = circ.copy()
        for index, q in enumerate(keep[0:-1]):
            if index % 2 == 0:
                circ_x_y_even.append(g1, [keep[index+1], keep[index]])
                #add_givens_rotation(circ_x_y_even, [keep[index+1], keep[index]])
            else:
                circ_x_y_odd.append(g1, [keep[index+1], keep[index]])
                #add_givens_rotation(circ_x_y_odd, [keep[index+1], keep[index]]) 
        # for j, q in enumerate(keep_aux):
        #     circ_x_y_even.measure(q, j)
        #     circ_x_y_odd.measure(q, j)
        circ_x_y_even.measure(keep_aux, list(range(len(keep_aux)))[::-1])
        circ_x_y_odd.measure(keep_aux, list(range(len(keep_aux)))[::-1])

        # circ_z = device_backend.get_compiled_circuit(circ_z, optimisation_level=2)
        # circ_x = device_backend.get_compiled_circuit(circ_x, optimisation_level=2)
        # circ_y = device_backend.get_compiled_circuit(circ_y, optimisation_level=2)
        # print('depth quantinuum', circ_z.depth())
        # print('1q gates quantinuum', circ_z.n_1qb_gates())
        # print('2q gates quantinuum', circ_z.n_2qb_gates())

            
        num_iter = 3 #int((self.N-self.N%3)/3) - 1

        output_list = []
        new_indices_list = []
        for i in range(num_iter):
            if i == 0:
                output = self.create_repeated_string()
            else:
                output = 'x'+output[:self.N-1]
            indices = [i for i, char in enumerate(output) if char == 'z']
            new_indices = []
            for j, index in enumerate(indices):
                if index != 2 and index != self.N - 3:
                    if output[indices[j]:indices[j]+4] == 'zxxz':
                        new_indices.append(index)
                elif index == 2:
                    if output[0:indices[j]+1] == 'xxz':
                        new_indices.append('left_b')
                        new_indices.append(index)
                elif index == self.N - 3:
                    if output[indices[j]:] == 'zxx':
                        #new_indices.append(index)
                        new_indices.append('right_b')

            output_list.append(output)
            new_indices_list.append(new_indices)

        circ_xy_list = []
        for string in output_list:
            circ_zxxz_zyyz = circ.copy()
            for j, q in enumerate(keep[0:-1]):
                if string[j] == 'x' and string[j+1] == 'x':
                    circ_zxxz_zyyz.append(g1, [keep[j+1], keep[j]])
                    #add_givens_rotation(circ_zxxz_zyyz, [keep[j+1], keep[j]])
            # for j, q in enumerate(keep_aux):
            #         circ_zxxz_zyyz.measure(q, j)
            circ_zxxz_zyyz.measure(keep_aux, list(range(len(keep_aux)))[::-1])

            circ_xy_list.append(circ_zxxz_zyyz)


        # Execute all circuits in one batch

        circs = [circ_z, circ_x_y_even, circ_x_y_odd] + circ_xy_list 

        #optimization_level = 3
        results, compiled_circuits = compile_ionq(circs, optimization_level, nshots, device, compile, counts=False)

        counts_z = results.get_counts(circs[0])
        counts_x_y_even = results.get_counts(circs[1])
        counts_x_y_odd = results.get_counts(circs[2])

        counts_z = counts_to_qibo(counts_z)
        counts_x_y_even = counts_to_qibo(counts_x_y_even)
        counts_x_y_odd = counts_to_qibo(counts_x_y_odd)
            
        counts_zxxz_zyyz_list = []
        for i in range(num_iter):
            counts_zxxz_zyyz = results.get_counts(circs[i+3])
            counts_zxxz_zyyz = counts_to_qibo(counts_zxxz_zyyz)
            counts_zxxz_zyyz_list.append(counts_zxxz_zyyz )

        return counts_x_y_even, counts_x_y_odd, counts_z, [new_indices_list, counts_zxxz_zyyz_list], compiled_circuits 
    

    def sample_q1(self, counts_z, boundaries):
        q1 = self.get_q1(boundaries)
        q1_val = q1.expectation_from_samples(counts_z)

        return q1_val

    def sample_q2(self, counts_z, boundaries):

        q2 = self.get_q2(boundaries)
        q2_val = q2.expectation_from_samples(counts_z)

        return q2_val
    
    def sample_ej(self, j, counts_z, boundaries):

        ej = self.get_ej(j, boundaries)
        ej_val = ej.expectation_from_samples(counts_z)

        return ej_val
    
    def sample_nonlocal_pauli(self, num, counts_z, boundaries):

        nonlocal_pauli = self.get_nonlocal_pauli(num, boundaries)
        nonlocal_pauli_val = nonlocal_pauli.expectation_from_samples(counts_z)

        return nonlocal_pauli_val
    
    def trace_frequencies(self, freqs, qubits):
        nqubits = len(list(freqs.keys())[0])
        freq_array = np.zeros(2**nqubits)
        for key, value in freqs.items():
            freq_array[int(key, 2)] = value

        #backend = NumpyBackend()
        unmeasured_qubits = tuple(i for i in range(nqubits) if i not in qubits)
        freq_array = np.reshape(freq_array, nqubits * (2,))
        freq_array = np.sum(freq_array, axis=unmeasured_qubits)
        freq_array = self.backend._order_probabilities(freq_array, qubits, nqubits).ravel()

        from collections import Counter

        freqs = Counter()
        for j in range(2**len(qubits)):
            if freq_array[j]!= 0:
                freqs[bin(j)[2:].zfill(len(qubits))] = int(freq_array[j])

        return freqs

    def sample_energy(self, counts_x, counts_y, new_indices_list, counts_zxxz_list, counts_zyyz_list, backend=None): # it only works without boundaries

        backend = _check_backend(backend)
 
        xx_yy = 0
        # if boundaries:
        #     for j in range(self.N-1):
        #         xx_yy += Z(j+1)*Z(j+2)
        # else:
        for j in range(self.N-1):
            xx_yy += Z(j)*Z(j+1)
        #xx_yy += Z(0)*Z(1)
        xx_yy = SymbolicHamiltonian(xx_yy, backend=self.backend)
        xx = xx_yy.expectation_from_samples(counts_x)
        yy = xx_yy.expectation_from_samples(counts_y)


        ##########################
        xx = 0
        yy = 0
        xx_yy_obs = backend.cast(np.kron(np.array([[1,0],[0,-1]]), np.array([[1,0],[0,-1]])))
        xx_yy_ham = Hamiltonian(2, xx_yy_obs, backend=self.backend)
        for j in range(0,self.N-1):
        #for j in [3]:
            counts_x_reduced = self.trace_frequencies(counts_x, [j,j+1])
            xx += xx_yy_ham.expectation_from_samples(counts_x_reduced)
            #print(xx_yy_ham.expectation_from_samples(counts_x_reduced), [j,j+1])

            counts_y_reduced = self.trace_frequencies(counts_y, [j,j+1])
            yy += xx_yy_ham.expectation_from_samples(counts_y_reduced)
            #print(xx_yy_ham.expectation_from_samples(counts_y_reduced), [j,j+1])
            #print(xx+yy)
        ####################################

        zxxz = 0
        zyyz = 0

        for k, indexes in enumerate(new_indices_list):
            counts_zxxz = counts_zxxz_list[k]
            counts_zyyz = counts_zyyz_list[k]
            for j  in indexes:
                if j == 'left_b':
                    zxxz_zyyz = Z(0)*Z(1)*Z(2)   
                    qubits = [0,1,2]                 
                elif j == 'right_b':
                    zxxz_zyyz = Z(0)*Z(1)*Z(2)
                    qubits = [self.N-3,self.N-2,self.N-1]
                else:
                    zxxz_zyyz = Z(0)*Z(1)*Z(2)*Z(3)
                    qubits = [j,j+1,j+2,j+3]

                zxxz_zyyz = SymbolicHamiltonian(zxxz_zyyz, backend=self.backend)
                
                counts_zxxz_j = self.trace_frequencies(counts_zxxz, qubits)

                counts_zyyz_j = self.trace_frequencies(counts_zyyz, qubits)

                zxxz += zxxz_zyyz.expectation_from_samples(counts_zxxz_j)
                zyyz += zxxz_zyyz.expectation_from_samples(counts_zyyz_j)


        energy = (-1/8)*(zxxz+zyyz+xx+yy)
        #energy = (-1/8)*(xx+yy)
        return energy
    
    # def test_exp_xx_yy(self,counts,qubits):
    #     nshots = sum(counts.values())
    #     xx_yy = 0
    #     for key, value in counts.items():
    #         if key[qubits[0]] == '0' and key[qubits[1]] == '1':
    #             xx_yy += -2*value
    #         elif key[qubits[0]] == '1' and key[qubits[1]] == '0':
    #             xx_yy += 2*value

    #     return xx_yy/nshots

    def sample_energy_postq2(self, counts_x_y_even, counts_x_y_odd, new_indices_list, counts_zxxz_zyyz_list, backend=None): # it only works without boundaries

        backend = _check_backend(backend)

        zero_zero = backend.cast(np.array([[1,0],[0,0]]))
        one_one = backend.cast(np.array([[0,0],[0,1]]))
        s = 1/np.sqrt(2)
        xx_yy_obs = backend.cast(np.array([
                [0, 0, 0, 0],
                [0, -2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 0]
            ]))

 
        xx_yy_even = 0
        xx_yy_odd = 0
        # if boundaries:
        #     for j in range(self.N-1):
        #         xx_yy += Z(j+1)*Z(j+2)
        # else:
        xx_yy_ham = Hamiltonian(2, xx_yy_obs, backend=self.backend)
        for j in range(0,self.N-1):
        #for j in [3]:
            if j % 2 == 0:
                counts_x_y_even_reduced = self.trace_frequencies(counts_x_y_even, [j,j+1])
                xx_yy_even += xx_yy_ham.expectation_from_samples(counts_x_y_even_reduced)
                #print(xx_yy_ham.expectation_from_samples(counts_x_y_even_reduced), [j,j+1])
                #print(xx_yy_even)
                #xx_yy_even = self.test_exp_xx_yy(counts_x_y_even, [j,j+1])
            else:
                counts_x_y_odd_reduced = self.trace_frequencies(counts_x_y_odd, [j,j+1])
                xx_yy_odd += xx_yy_ham.expectation_from_samples(counts_x_y_odd_reduced)
                #print(xx_yy_ham.expectation_from_samples(counts_x_y_odd_reduced), [j,j+1])

                #xx_yy_odd = self.test_exp_xx_yy(counts_x_y_odd, [j,j+1])
            #print(xx_yy_even + xx_yy_odd, [j,j+1])
        zxxz_zyyz = 0
        z_matrix = backend.cast(np.array([[1,0],[0,-1]]))
        for k, indexes in enumerate(new_indices_list):
            counts_zxxz_zyyz = counts_zxxz_zyyz_list[k]
            for j  in indexes:
                if j == 'left_b': 
                    zxxz_zyyz_mat = backend.np.kron(xx_yy_obs, z_matrix)
                    qubits = [0,1,2]
                    zxxz_zyyz_ham  = Hamiltonian(3, zxxz_zyyz_mat, backend=self.backend)                 
                elif j == 'right_b':
                    zxxz_zyyz_mat = backend.np.kron(z_matrix, xx_yy_obs)
                    qubits = [self.N-3,self.N-2,self.N-1]
                    zxxz_zyyz_ham  = Hamiltonian(3, zxxz_zyyz_mat, backend=self.backend)
                else:
                    zxxz_zyyz_mat = backend.np.kron(z_matrix, backend.np.kron(xx_yy_obs, z_matrix))
                    qubits = [j,j+1,j+2,j+3]
                    zxxz_zyyz_ham  = Hamiltonian(4, zxxz_zyyz_mat, backend=self.backend)            
                
                counts_zxxz_zyyz_j = self.trace_frequencies(counts_zxxz_zyyz, qubits)

                zxxz_zyyz += zxxz_zyyz_ham.expectation_from_samples(counts_zxxz_zyyz_j)


        energy = (-1/8)*(zxxz_zyyz+xx_yy_even+xx_yy_odd)
        #energy = (-1/8)*(xx_yy_even+xx_yy_odd)
        return energy