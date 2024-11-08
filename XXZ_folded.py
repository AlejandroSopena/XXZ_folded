import numpy as np

from qibo import gates, Circuit
from qibo.symbols import X, Y, Z
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.quantum_info import fidelity, partial_trace
from qibo.backends import GlobalBackend, construct_backend
import cupy as cp

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
    cp.get_default_memory_pool().free_all_blocks()

    return result


class XXZ_folded_one_domain:
    """
    A class to build the circuits to prepare the eigenstates of the XXZ folded with two domain walls.

    Args:
        N (int): The number of bulk qubits.
        M (int): The number of domain walls.
        domain_pos (list): The positions of the domain walls.
    """
    def __init__(self, N=8, M=1, D=2, domain_pos=[[5, 6, 7]]):
        self.N = N
        self.M = M
        self.D = D
        self.domain_pos = domain_pos

    def _get_roots(self):
        roots = []
        for i in range(self.M):
            p = (i+1)*np.pi/(self.N+2-self.M-self.D)
            roots.append(p)
        self.roots = roots

    def get_b_circuit(self):
        b_circuit = get_b_circuit(self.N-self.D-self.M+1, self.M, self.roots)

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
        for q in reversed(range(self.N-self.D-self.M+1)):
            for j in range(self.M):
                circ_u0.add(gates.SWAP(self.N-self.D + aux_qubits1 - 2 - j,
                            self.N-self.D + aux_qubits1 - 1 - j).controlled_by(q))
            for j in range(self.M-1):
                if q < q+self.M-1-j:
                    circ_u0.add(gates.SWAP(q, q+self.M-1 -
                                j).controlled_by(self.N-self.D+1+j))
        self.circ_u0 = circ_u0

        return circ_u0

    def get_D_circ_general(self):
        aux = 2 + int(self.D/2) + 1 + int(self.D/2) + 1
        nqubits_d = 2*self.N - self.D
        r_r = list(range(nqubits_d+aux-(int(self.D/2) +1),nqubits_d+aux))
        r_c = list(range(nqubits_d+aux-2*(int(self.D/2) +1),nqubits_d+aux-int(self.D/2) -1))
        r_0 = list(range(nqubits_d+aux-2*(int(self.D/2) + 1)-2,nqubits_d+aux-2*(int(self.D/2) + 1))) 
        #[phys,r0,rc,rr]
        circ_d = Circuit(nqubits_d+aux)
        print(aux)
        circ_d.add(gates.X(r_c[0]))
        circ_d.add(gates.X(r_r[0]))
        nqubits = circ_d.nqubits
        # for domain in self.domain_pos:
        #     index_domain = []

        #     i = 1
        #     for j in domain:
        #         if j <= self.N - self.D:
        #             index_domain.append(2*j-1)
        #         else:
        #             index_domain.append(2*(self.N-self.D)-1+i)
        #             i += 1
        #      circ_d.add([gates.X(index) for index in index_domain])  # define domain

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

        index_domain = []
        k = 1
        for j in range(0, self.N - self.D):
            index_domain.append(k)
            k += 2
        k -= 1
        for j in range(self.N - self.D, self.N):
            index_domain.append(k)
            k += 1
        
        #circ_d.add(gates.X(index_p[0])) # ADD MAGNON #
        #print(index_p[3]
        for p in reversed(index_p):
        #for p in [index_p[5],index_p[4],index_p[3],index_p[2],index_p[1],index_p[0]]:#reversed(index_p):#,index_p[1],index_p[0]]:
        #for p in [index_p[0]]:
        #p = index_p[3]

            circ_d.add(gates.SWAP(p, r_0[0]))
            # Detect and move ni
            print('p',p)
            if p >= 2:
                index_q = [-1] + index_domain
                index_q_loop = [-1] + index_domain[0:index_p.index(p)-1]
                for q in index_q_loop:
                    index_q1 = index_q.index(q)
                    circ_d.add(gates.X(index_q[index_q1+2]))
                    circ_d.add(gates.X(
                        r_0[1]).controlled_by(index_q[index_q1+2], index_q[index_q1+3], r_0[0]))
                    circ_d.add(gates.X(index_q[index_q1+2]))

                    circ_d.add(gates.CNOT(r_0[1], index_q[index_q1+1]))
                    circ_d.add(gates.CNOT(r_0[1], index_q[index_q1+2]))
                    for ii in reversed(range(1,len(r_c))):
                        circ_d.add(gates.SWAP(r_c[ii-1], r_c[ii]).controlled_by(r_0[1]))

                    if q == -1:
                        circ_d.add(
                            gates.X(r_0[1]).controlled_by(index_q[index_q1+1], r_0[0]))
                    else:
                        circ_d.add(gates.X(q))
                        circ_d.add(gates.X(r_0[1]).controlled_by(q,
                                    index_q[index_q1+1], r_0[0]))
                        circ_d.add(gates.X(q))

            i2 = index_p.index(p)
            for index_q, q in enumerate(index_domain[1:i2]): #no +2 ############ CHANGE
                # Detect and move nf
                circ_d.add(gates.X(index_domain[1+index_q+3]))  # 6))
                circ_d.add(gates.X(
                    r_0[1]).controlled_by(index_domain[1+index_q+2], index_domain[1+index_q+3], r_0[0]))
                circ_d.add(gates.X(index_domain[1+index_q+3]))

                circ_d.add(gates.CNOT(r_0[1], index_domain[1+index_q+1]))
                circ_d.add(gates.CNOT(r_0[1], index_domain[1+index_q+2]))

                circ_d.add(gates.X(index_domain[1+index_q+1]))
                circ_d.add(gates.X(r_0[1]).controlled_by(q,
                            index_domain[1+index_q+1], r_0[0]))
                circ_d.add(gates.X(index_domain[1+index_q+1]))

            #insert magnon
            ip = index_p.index(p)
            circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[ip], r_0[0]))
            circ_d.add(gates.X(index_domain[ip]).controlled_by(r_0[0],r_c[0]))
            for i, ii in enumerate(index_domain[ip+1:ip+self.D:2]):
                circ_d.add(gates.X(ii).controlled_by(r_0[1],r_c[i+1]))
                circ_d.add(gates.X(r_0[1]))
                circ_d.add(gates.X(index_domain[index_domain.index(ii)+1]).controlled_by(r_0[1],r_c[i+1]))
                circ_d.add(gates.X(r_0[1]))
            if ip >= 1:
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[ip-1],index_domain[ip],r_0[0]))

            # #reset
            #if p != index_p[0]:
            ip = index_p.index(p)
            #print(ip,self.N-self.D-1)
            if ip == 0:
                circ_d.add(gates.X(index_domain[ip+1]))
                circ_d.add(gates.X(r_0[0]).controlled_by(index_domain[ip], index_domain[ip+1]))
                # circ_d.add(gates.CNOT(r_0[1],r_0[0]))
                # circ_d.add(gates.CNOT(r_0[1],r_c[0]))
                # circ_d.add(gates.CNOT(r_0[1],r_c[1]))
                #circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[ip]))
                circ_d.add(gates.X(index_domain[ip+1]))
                #circ_d.add(gates.X(r_0[1])) #############
            else:
                circ_d.add(gates.X(index_domain[ip-1]))
                circ_d.add(gates.X(index_domain[ip+1]))
                circ_d.add(gates.X(r_0[0]).controlled_by(index_domain[ip-1], index_domain[ip], index_domain[ip+1]))
                circ_d.add(gates.X(index_domain[ip-1]))
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[ip-1], index_domain[ip], index_domain[ip+1],index_domain[ip+2]))
                circ_d.add(gates.CNOT(r_0[1],r_0[0]))
                circ_d.add(gates.CNOT(r_0[1],r_c[0]))
                circ_d.add(gates.CNOT(r_0[1],r_c[1]))
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[ip-1], index_domain[ip], index_domain[ip+1],index_domain[ip+2]))
                circ_d.add(gates.X(index_domain[ip+1]))
                #####
                ll = int(min(ip,4))
                index_start = 1#int(min(1,ip-3))
                print('aa',index_start)
                for ii in range(index_start,ip): #range(ip-ll,ip):#range(index_start,ip):#range(ip-ll,ip):
                    circ_d.add(gates.X(index_domain[ii+1]))
                    for jj in reversed(range(1,len(r_r))):
                        circ_d.add(gates.SWAP(r_r[jj-1], r_r[jj]).controlled_by(index_domain[ii],index_domain[ii+1]))
                    circ_d.add(gates.X(index_domain[ii+1]))

            
            for index_iip,iip in enumerate(range(ip+2,ip+self.D,2)):
                circ_d.add(gates.X(index_domain[iip-1]))
                circ_d.add(gates.X(index_domain[iip+1]))
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[iip-1], index_domain[iip], index_domain[iip+1]))
                circ_d.add(gates.X(r_0[0]).controlled_by(r_r[index_iip+1],r_0[1]))
                circ_d.add(gates.X(r_c[0]).controlled_by(r_r[index_iip+1],r_0[1]))
                circ_d.add(gates.X(r_c[index_iip+1]).controlled_by(r_r[index_iip+1],r_0[1]))
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[iip-1], index_domain[iip], index_domain[iip+1]))
                circ_d.add(gates.X(index_domain[iip-1]))
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[iip-1], index_domain[iip], index_domain[iip+1],index_domain[iip+2]))
                circ_d.add(gates.X(r_0[0]).controlled_by(r_r[index_iip+1],r_0[1]))
                circ_d.add(gates.X(r_c[0]).controlled_by(r_r[index_iip+1],r_0[1]))
                circ_d.add(gates.X(r_c[index_iip+2]).controlled_by(r_r[index_iip+1],r_0[1]))
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[iip-1], index_domain[iip], index_domain[iip+1],index_domain[iip+2]))
                circ_d.add(gates.X(index_domain[iip+1]))
                if ip != 0:
                    index_start = iip - 2
                    for ii in range(index_start,iip): #range(iip-2,iip):
                        circ_d.add(gates.X(index_domain[ii+1]))
                        for jj in reversed(range(1,len(r_r))):
                            circ_d.add(gates.SWAP(r_r[jj-1], r_r[jj]).controlled_by(index_domain[ii],index_domain[ii+1]))
                        circ_d.add(gates.X(index_domain[ii+1]))

            iip = ip+self.D
            #index_iip = int(self.D/2) - 1
            circ_d.add(gates.X(index_domain[iip-1]))
            print(ip,self.N-self.D-1)
            if ip < self.N-self.D-1:
                circ_d.add(gates.X(index_domain[iip+1]))
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[iip-1], index_domain[iip], index_domain[iip+1]))
            else:
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[iip-1], index_domain[iip]))
            circ_d.add(gates.X(r_0[0]).controlled_by(r_r[-1],r_0[1]))
            circ_d.add(gates.X(r_c[0]).controlled_by(r_r[-1],r_0[1]))
            circ_d.add(gates.X(r_c[-1]).controlled_by(r_r[-1],r_0[1]))
            if ip < self.N-self.D-1:
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[iip-1], index_domain[iip], index_domain[iip+1]))
                circ_d.add(gates.X(index_domain[iip+1]))
            else:
                circ_d.add(gates.X(r_0[1]).controlled_by(index_domain[iip-1], index_domain[iip]))
            circ_d.add(gates.X(index_domain[iip-1]))
            ll = int(min(ip,3))
            index_start = 1#int(max(1,ip-3))
            for ii in reversed(range(index_start,ip+self.D-2)):#reversed(range(ip-ll,ip+self.D-2)):#reversed(range(index_start,ip+self.D-2)):#reversed(range(ip-ll,ip+self.D-2)):
                circ_d.add(gates.X(index_domain[ii+1]))
                for jj in range(1,len(r_r)):
                    circ_d.add(gates.SWAP(r_r[jj-1], r_r[jj]).controlled_by(index_domain[ii],index_domain[ii+1]))
                circ_d.add(gates.X(index_domain[ii+1]))

        self.circ_d = circ_d
        sym_state = circ_d().symbolic()
        sym_state = sym_state[7:-1]
        new_state = ''.join([sym_state[i] for i in index_domain])
        print(new_state)
        print(''.join([sym_state[i] for i in r_0+r_c+r_r]))

        return circ_d

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

        circ_d.add(gates.SWAP(index_p[2], nqubits-3))

        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
        circ_d.add(gates.CNOT(index_domain[0], nqubits-2))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[0], nqubits-3))
        circ_d.add(gates.X(index_domain[0]))

        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[4], nqubits-3))
        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(gates.X(index_domain[2]))

        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))     
        circ_d.add(gates.CNOT(nqubits-1, index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))     
        circ_d.add(gates.X(index_domain[2]))

        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.CNOT(index_domain[3], nqubits-3))
        circ_d.add(gates.CNOT(index_domain[3], nqubits-1))
        circ_d.add(gates.X(index_domain[3]))

        circ_d.add(gates.SWAP(index_p[1], nqubits-3))

        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[0], nqubits-3))

        circ_d.add(gates.CNOT(nqubits-3, index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-1, index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-1, index_domain[1]))

        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.X(
            nqubits-3).controlled_by(index_domain[0], index_domain[1], index_domain[2]))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.X(nqubits-3).controlled_by(
            index_domain[0], index_domain[1], index_domain[2], index_domain[3]))
        circ_d.add(gates.X(nqubits-1).controlled_by(
            index_domain[0], index_domain[1], index_domain[2], index_domain[3]))
        circ_d.add(gates.X(index_domain[2]))

        circ_d.add(gates.SWAP(index_p[0], nqubits-3))

        circ_d.add(gates.CNOT(nqubits-3, index_domain[0]))

        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(
            gates.X(nqubits-3).controlled_by(index_domain[0], index_domain[1]))
        circ_d.add(gates.X(index_domain[1]))

        self.circ_d = circ_d

        return circ_d

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

        #circ_d.add(gates.X(index_p[1])) # ADD MAGNON

        circ_d.add(gates.SWAP(index_p[3], nqubits-3))

        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
        circ_d.add(gates.CNOT(index_domain[0], nqubits-2))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[2], index_domain[3], nqubits-3))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], index_domain[3], nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[0], index_domain[1], nqubits-3))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[3], nqubits-3))
        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[3], nqubits-3))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[1], nqubits-3))
        circ_d.add(gates.X(index_domain[1]))

        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[4], nqubits-3))
        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.X(index_domain[5]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[4], index_domain[5], nqubits-3))
        circ_d.add(gates.X(index_domain[5]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], index_domain[3]))
        circ_d.add(gates.X(index_domain[3]))

        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[3], nqubits-3))
        circ_d.add(gates.CNOT(nqubits-1, index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[5]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[3], nqubits-3))
        circ_d.add(gates.X(index_domain[3]))

        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.X(
            nqubits-3).controlled_by(index_domain[4], index_domain[5]))
        circ_d.add(gates.X(
            nqubits-1).controlled_by(index_domain[4], index_domain[5]))
        circ_d.add(gates.X(index_domain[4]))

        circ_d.add(gates.SWAP(index_p[2], nqubits-3))

        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
        #circ_d.add(gates.CNOT(index_domain[0],nqubits-2))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[0], nqubits-3)) #NEW
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[2], index_domain[3], nqubits-3))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], index_domain[3], nqubits-3))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add( #NEW
            gates.X(nqubits-2).controlled_by(index_domain[0], index_domain[1], nqubits-3)) #NEW
        circ_d.add(gates.X(index_domain[0]))

        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[4], nqubits-3))
        circ_d.add(gates.X(index_domain[4]))
        circ_d.add(gates.CNOT(nqubits-3, index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-1, index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-1, index_domain[3]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
        # circ_d.add(gates.CNOT(nqubits-2, index_domain[2]))
        # circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
        # circ_d.add(gates.X(index_domain[2]))
        # circ_d.add(
        #     gates.X(nqubits-2).controlled_by(index_domain[1], index_domain[2], nqubits-3))
        
        # circ_d.add(
        #     gates.X(nqubits-2).controlled_by(index_domain[1], index_domain[2], nqubits-3))
        # circ_d.add(gates.X(index_domain[2]))
        # circ_d.add(gates.CNOT(nqubits-3, index_domain[2]))
        # circ_d.add(gates.CNOT(nqubits-1, index_domain[2]))
        # circ_d.add(gates.CNOT(nqubits-1, index_domain[3]))
        # circ_d.add(gates.CNOT(nqubits-2, index_domain[3]))
        # circ_d.add(gates.CNOT(nqubits-2, index_domain[4]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[1], index_domain[2], nqubits-3))
        circ_d.add(gates.X(index_domain[2]))


        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(gates.X(
            nqubits-3).controlled_by(index_domain[1], index_domain[2], index_domain[3]))
        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(gates.X(index_domain[3]))
        circ_d.add(gates.X(nqubits-3).controlled_by(
            index_domain[1], index_domain[2], index_domain[4]))
        circ_d.add(gates.X(nqubits-1).controlled_by(
            index_domain[1], index_domain[2], index_domain[4]))
        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.X(
            nqubits-3).controlled_by(index_domain[1], index_domain[2], index_domain[4]))
        circ_d.add(gates.X(
            nqubits-1).controlled_by(index_domain[1], index_domain[2], index_domain[4]))
        circ_d.add(gates.X(index_domain[2]))
        
        # circ_d.add(gates.X(index_domain[3]))
        # circ_d.add(gates.X(index_domain[1]))
        # circ_d.add(gates.X(
        #     nqubits-3).controlled_by(index_domain[1], index_domain[2], index_domain[3]))
        # circ_d.add(gates.X(index_domain[3]))
        # circ_d.add(gates.X(index_domain[1]))
        # # circ_d.add(
        # #     gates.X(nqubits-1).controlled_by(index_domain[1], index_domain[4]))
        # # circ_d.add(
        # #     gates.X(nqubits-3).controlled_by(index_domain[1], index_domain[4]))
        # circ_d.add(gates.CNOT(index_domain[1], nqubits-1))
        # circ_d.add(gates.CNOT(index_domain[1], nqubits-3)) #HERE FAIL




        circ_d.add(gates.SWAP(index_p[1], nqubits-3))

        circ_d.add(
            gates.X(nqubits-1).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[2], nqubits-3))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[0]))
        circ_d.add(gates.CNOT(nqubits-2, index_domain[1]))
        circ_d.add(
            gates.X(nqubits-2).controlled_by(index_domain[0], nqubits-3))

        circ_d.add(gates.CNOT(nqubits-3, index_domain[1]))
        circ_d.add(gates.CNOT(nqubits-1, index_domain[2]))
        circ_d.add(gates.CNOT(nqubits-1, index_domain[1]))

        circ_d.add(gates.X(index_domain[2]))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.X(
            nqubits-3).controlled_by(index_domain[0], index_domain[1], index_domain[2]))
        circ_d.add(gates.X(index_domain[0]))
        circ_d.add(gates.X(
            nqubits-3).controlled_by(index_domain[0], index_domain[2], index_domain[3]))
        circ_d.add(gates.X(
            nqubits-1).controlled_by(index_domain[0], index_domain[2], index_domain[3]))
        circ_d.add(gates.X(index_domain[2]))

        circ_d.add(gates.SWAP(index_p[0], nqubits-3))

        circ_d.add(gates.CNOT(nqubits-3, index_domain[0]))

        circ_d.add(gates.X(index_domain[1]))
        circ_d.add(
            gates.X(nqubits-3).controlled_by(index_domain[0], index_domain[1]))
        circ_d.add(gates.X(index_domain[1]))

        self.circ_d = circ_d

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
            aux = 2 + int(self.D/2) + 1 + int(self.D/2) + 1
        circ_full = Circuit(2*self.N-self.D+aux+aux_qubits1)

        index_p = []
        k = 0
        for j in range(0, self.N - self.D):
            index_p.append(k)
            k += 2

        new_qubits_1 = index_p + \
            list(range(2*self.N-self.D, 2*self.N-self.D+aux_qubits1))
        new_qubits_2 = list(range(0, 2*self.N-self.D)) + list(range(2 *
                                                                    self.N-self.D+aux_qubits1, 2*self.N-self.D+aux_qubits1+aux))

        circ_full.add(circ1.on_qubits(*new_qubits_1))
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
        ham = SymbolicHamiltonian(ham)

        return ham

    def get_q1(self, boundaries=True):
        q1 = 0
        if boundaries:
            for j in range(self.N+2):
                q1 += (1/2)*(1-Z(j))
        else:
            for j in range(0, self.N):
                q1 += (1/2)*(1-Z(j))
        q1 = SymbolicHamiltonian(q1)

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

        q2 = SymbolicHamiltonian(q2)

        return q2

    def get_state(self, noise_model=None, boundaries=True, density_matrix=False, state=None, layout=None, backend=None):
        if backend is None:
            backend = GlobalBackend()

        circ = self.circ_full
        if noise_model is not None:
            circ = noise_model.apply(circ)
        circ.density_matrix = density_matrix
        if state is None:
            result = backend.execute_circuit(circ)  # circ()
            state1 = result.state()
        else:
            state1 = state

        if boundaries:
            if (self.N == 5 and self.M == 1 and self.D == 2) or (self.N == 6 and self.M == 1 and self.D == 2):
                keep = [self.circ_full.nqubits-1]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                    self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-2]
            else:
                keep = [self.circ_full.nqubits-2-(int(self.D/2) + 1 + int(self.D/2) + 1)]+[2*j+1 for j in range(self.N-self.D)] + [2*(
                    self.N-self.D)+i for i in range(self.D)]+[self.circ_full.nqubits-1-(int(self.D/2) + 1 + int(self.D/2) + 1)]
        else:
            if self.M == 1:
                keep = [2*j+1 for j in range(self.N-self.D)] + \
                    [2*(self.N-self.D)+i for i in range(self.D)]
            else:
                keep = [2*j+1 for j in range(self.N-self.D)] + \
                    [2*(self.N-self.D)+i for i in range(self.D)]

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
            #print(np.where(abs(ham_state)>1e-10))
            ham_state = ham_state / np.linalg.norm(ham_state)
        fid_ham = fidelity(ham_state, state)

        if dm:
            q1_state = q1@state@q1.conjugate().transpose()
            q1_state = q1_state / np.trace(q1_state)
        else:
            q1_state = q1@state
            q1_state = q1_state / np.linalg.norm(q1_state)
        fid_q1 = fidelity(q1_state, state)

        if dm:
            q2_state = q2@state@q2.conjugate().transpose()
            q2_state = q2_state / np.trace(q2_state)
        else:
            q2_state = q2@state
            q2_state = q2_state / np.linalg.norm(q2_state)
        fid_q2 = fidelity(q2_state, state)

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

        if backend is None:
            backend = GlobalBackend()

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

        if backend is None:
            backend = GlobalBackend()

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
