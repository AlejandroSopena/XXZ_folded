import numpy as np

from qibo import gates, Circuit
from qibo.symbols import X, Y, Z
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.quantum_info import fidelity, partial_trace
from qibo.backends import _check_backend, construct_backend

from initial_b_matrix import get_b_circuit
from XX_model import XX_model


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
    def __init__(self, N=8, M=1, D=2, domain_pos=[[5, 6, 7]], backend=None):
        self.N = N
        self.M = M
        self.D = D
        self.domain_pos = domain_pos
        self.backend = _check_backend(backend)

    def _get_roots(self):
        roots = []
        for i in range(self.M):
            p = (i+1)*np.pi/(self.N+2-self.M-self.D)
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

    def get_state(self, noise_model=None, boundaries=True, density_matrix=False, state=None, layout=None):
        if self.D == 0:
            circ = self.circ_Psi_M_0
        else:
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

    def circ_to_qiskit(self, circ):
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import XGate, SwapGate, CXGate, UnitaryGate, CZGate

        backend = construct_backend('numpy')
        gate_list = circ.queue
        circ_qiskit = QuantumCircuit(circ.nqubits, self.N)
        for g in gate_list:
            control_qubits = g.control_qubits[::-1]
            target_qubits = g.target_qubits[::-1]
            if isinstance(g, gates.X) or isinstance(g, gates.SWAP) or isinstance(g, gates.TOFFOLI):
                if isinstance(g, gates.X):
                    g1 = XGate()
                elif isinstance(g, gates.SWAP):
                    g1 = SwapGate()
                elif isinstance(g, gates.TOFFOLI):
                    g1 = XGate()
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
                    
        return circ_qiskit
    
    def circ_to_quantinuum(self, circ):
        from pytket.circuit import Circuit, OpType, QControlBox, Unitary2qBox, Unitary1qBox, Op

        backend = construct_backend('numpy')
        gate_list = circ.queue
        circ_quantinuum = Circuit(circ.nqubits, self.N)
        for g in gate_list:
            control_qubits = g.control_qubits
            target_qubits = g.target_qubits
            if isinstance(g, gates.X) or isinstance(g, gates.SWAP) or isinstance(g, gates.TOFFOLI):
                if isinstance(g, gates.X):
                    g1 = OpType.X
                elif isinstance(g, gates.SWAP):
                    g1 = OpType.SWAP
                elif isinstance(g, gates.TOFFOLI):
                    g1 = OpType.X
                if len(control_qubits) == 0:
                    circ_quantinuum.add_gate(g1, target_qubits)
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

    def sample_energy(self, counts_x, counts_y, nshots, noise_model, layout, boundaries, backend=None):

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

        backend = _check_backend(backend)

        xx_yy = 0
        if boundaries:
            for j in range(self.N-1):
                xx_yy += Z(j+1)*Z(j+2)
        else:
            for j in range(self.N-2):
                xx_yy += Z(j+1)*Z(j+2)
            xx_yy += Z(0)*Z(1)
        xx_yy = SymbolicHamiltonian(xx_yy)
        xx = xx_yy.expectation_from_samples(counts_x)
        yy = xx_yy.expectation_from_samples(counts_y)

        zxxz = 0
        zyyz = 0

        if boundaries:
            vals = self.N-1
        else:
            vals = self.N-3

        for j in range(vals):
            zxxz_zyyz = Z(j)*Z(j+1)*Z(j+2)*Z(j+3)
            zxxz_zyyz = SymbolicHamiltonian(zxxz_zyyz)

            circ_x = circ.copy()
            circ_x.add(gates.H(keep[j+1]))
            circ_x.add(gates.H(keep[j+2]))
            circ_x.add(gates.M(*keep))

            circ_y = circ.copy()
            circ_y.add(gates.SDG(keep[j+1]))
            circ_y.add(gates.H(keep[j+1]))
            circ_y.add(gates.SDG(keep[j+2]))
            circ_y.add(gates.H(keep[j+2]))
            circ_y.add(gates.M(*keep))

            result_x = backend.execute_circuit(circ_x, nshots=nshots)
            counts_x = result_x.frequencies()

            result_y = backend.execute_circuit(circ_y, nshots=nshots)
            counts_y = result_y.frequencies()

            zxxz += zxxz_zyyz.expectation_from_samples(counts_x)
            zyyz += zxxz_zyyz.expectation_from_samples(counts_y)

        if boundaries is False:
            zxxz_zyyz = Z(0)*Z(1)*Z(2)
            zxxz_zyyz = SymbolicHamiltonian(zxxz_zyyz)

            circ_x = circ.copy()
            circ_x.add(gates.H(keep[0]))
            circ_x.add(gates.H(keep[1]))
            circ_x.add(gates.M(*keep))

            circ_y = circ.copy()
            circ_y.add(gates.SDG(keep[0]))
            circ_y.add(gates.H(keep[0]))
            circ_y.add(gates.SDG(keep[1]))
            circ_y.add(gates.H(keep[1]))
            circ_y.add(gates.M(*keep))

            result_x = backend.execute_circuit(circ_x, nshots=nshots)
            counts_x = result_x.frequencies()

            result_y = backend.execute_circuit(circ_y, nshots=nshots)
            counts_y = result_y.frequencies()

            zxxz += zxxz_zyyz.expectation_from_samples(counts_x)
            zyyz += zxxz_zyyz.expectation_from_samples(counts_y)

            zxxz_zyyz = Z(self.N-3)*Z(self.N-2)*Z(self.N-1)
            zxxz_zyyz = SymbolicHamiltonian(zxxz_zyyz)

            circ_x = circ.copy()
            circ_x.add(gates.H(keep[self.N-2]))
            circ_x.add(gates.H(keep[self.N-1]))
            circ_x.add(gates.M(*keep))

            circ_y = circ.copy()
            circ_y.add(gates.SDG(keep[self.N-2]))
            circ_y.add(gates.H(keep[self.N-2]))
            circ_y.add(gates.SDG(keep[self.N-1]))
            circ_y.add(gates.H(keep[self.N-1]))
            circ_y.add(gates.M(*keep))

            result_x = backend.execute_circuit(circ_x, nshots=nshots)
            counts_x = result_x.frequencies()

            result_y = backend.execute_circuit(circ_y, nshots=nshots)
            counts_y = result_y.frequencies()

            zxxz += zxxz_zyyz.expectation_from_samples(counts_x)
            zyyz += zxxz_zyyz.expectation_from_samples(counts_y)

        energy = (-1/8)*(zxxz+zyyz+xx+yy)

        return energy

    def sample_q1(self, counts_z, boundaries):
        q1 = self.get_q1(boundaries)
        q1_val = q1.expectation_from_samples(counts_z)

        return q1_val

    def sample_q2(self, counts_z, boundaries):

        q2 = self.get_q2(boundaries)
        q2_val = q2.expectation_from_samples(counts_z)

        return q2_val
