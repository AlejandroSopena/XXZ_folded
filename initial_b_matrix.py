import numpy as np
from scipy.optimize import basinhopping

from qibo import gates
from qibo.backends import _check_backend
from qibo.models import Circuit
from qibo.quantum_info import fidelity

def binarize(decimal, L):
    binary = np.zeros(L)
    bin = ''
    for i in range(L):
        binary[L - 1 - i] = decimal % 2
        decimal = decimal // 2
        bin = bin + str(binary.astype(int)[L - 1 - i])
    return bin[::-1]

def P_flip(P):
    rows=np.shape(P)[0]
    L1=int(np.log2(rows))
    P1=np.zeros(rows,dtype='complex')
    for i in range(rows):
        P1[int(binarize(i,L1)[::-1],2)]=P[i]
    return P1

def P_flip_mat(P):
    rows=np.shape(P)[0]
    L1=int(np.log2(rows))
    columns=np.shape(P)[1]
    L2=int(np.log2(columns))
    P1=np.zeros((rows, columns),dtype='complex')
    for i in range(rows):
        for j in range(columns):
            P1[int(binarize(i,L2)[::-1],2),int(binarize(j,L1)[::-1],2)]=P[i,j]
    return P1

class XXZ_free_open_model:
    def __init__(self, nqubits, nmagnons):
        self.nqubits = nqubits
        self.nmagnons = nmagnons
        self.roots = None
        self.circuit = None
        self.delta = 0

    def get_roots(self, moments):
        roots = []
        for l in range(self.nmagnons):
            roots.append(np.exp(1j*moments[l]))
            roots.append(np.exp(-1j*moments[l]))
        self.roots = np.array(roots)

    def m_k(self,n):
        return min(n+1,2*self.nmagnons)

    def choose(self,n, m):
        return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))

    def _get_index(self, r):
        def gen_l(l,j):
            l = np.array(l)
            r = len(l)
            l_list=[np.array(l)]
            if j != r-1:
                while l[j] < l[j+1] - 1:
                        l[j] += 1
                        l_list.append(np.array(l))

            else:
                while l[j] < 2*self.nmagnons-1:
                    l[j] += 1
                    l_list.append(np.array(l))

            return l_list

        l=list(range(r))
        j = len(l) - 1
        result1 = gen_l(l,j)
        for j in reversed(range(0,len(l) - 1)):
            r_list=[]
            for rr in result1:
                r_list.append(gen_l(rr,j))
            r_list = np.concatenate(r_list)
            result1 = r_list

        return result1

    def get_indexes(self):
        indexes = []
        for r in range(1,2*self.nmagnons+1):
            indexes.append(self._get_index(r))
        self.indexes = indexes

    def get_b(self, x, y):
        return x + 1/x + y + 1/y

    def Cr1_xy(self, r, k, x, y):
        if k == 0:
            c = 0
            return c
        c = self.Cr1_xy(r,k-1,x,y)*np.conjugate(x)*y + 1
        
        return c

    def Crk_ab(self, r, k, a, b):
        index = self.indexes[r-1]
        x = index[a]
        y = index[b]
        dim = len(x)
        c = np.zeros((dim,dim),complex)
        a1 = self.roots[x]
        b1 = self.roots[y]
        for n in range(dim):
            for m in range(dim):
                c[n,m] = self.Cr1_xy(r,k,a1[n],b1[m])

        return np.linalg.det(c)


    def _get_Crk(self, r, k):
        from joblib import Parallel, delayed
        j = int(self.choose(self.m_k(k), r))
        C = Parallel(n_jobs=30)(delayed(self.Crk_ab)(r, k, n, m) for n in range(j) for m in range(j))
        C = np.reshape(C,(j,j))

        return C

    def Ark(self, r, k):
        c = self._get_Crk(r,k)
        a = np.linalg.cholesky(c) 
        a = np.transpose(np.conjugate(a))

        return a

    def Ark(self,r,k):
        def choose(n, m):
            return np.math.factorial(n)/(np.math.factorial(m)*np.math.factorial(n-m))
        rows = int(choose(self.m_k(k), r))
        cols = int(choose(self.m_k(k), r))
        A = np.zeros((rows,cols),complex)
        c = self._get_Crk(r, k)
        for a in range(rows):
            for b in range(cols):
                c_a_to_b = np.copy(c)
                c_a_to_b[:,a] = c[:,b]
                if a == 0:
                    det_a = 1
                else:
                    det_a = np.linalg.det(c[0:a,0:a])
                det_aplus1 = np.linalg.det(c[0:a+1,0:a+1])
                det_aplus1_ab = np.linalg.det(c_a_to_b[0:a+1,0:a+1])
                A[a,b] = det_aplus1_ab / np.sqrt(det_a*det_aplus1)

        return A
   
    def get_v(self):
        s1 = np.array([0,0,1,0],complex)
        s1 = np.reshape(s1,(-1,1))
        s2 = np.array([0,1,0,0],complex)
        s2 = np.reshape(s2,(-1,1))
        v = self.roots[0]*s1 - self.roots[1]*s2
        k = 2
        for l in range(1,self.nmagnons):
            v = np.kron(v,self.roots[k]*s1 - self.roots[k+1]*s2)
            k += 2
        v = P_flip(v)
        
        return np.reshape(v,(-1,1))

    def get_u(self):
        u = np.zeros(2**(2*self.nmagnons),complex)
        v = self.get_v()
        A = self.Ark(self.nmagnons,self.nqubits)
        elements = []
        for j in range(2**(2*self.nmagnons)):
            if bin(j).count('1') == self.nmagnons:
                elements.append(j)
        for a in range(len(elements)):
            vec = np.zeros(2**(2*self.nmagnons),complex)
            vec[elements[a]] = 1
            for b in range(len(elements)):
                u += A[a,b]*v[elements[b]]*vec

        return u

def diagonal(parameters):
    """
    Diagonal matrix providing gauge-freedom.

    Args:
        parameters (:array:float): 
    Returns:
        (np.array):
    """
    dim = len(parameters)        
    d = np.zeros(shape=(dim,dim), dtype=np.complex128)

    for i,p in enumerate(parameters):
        d[i,i] = np.e**(1j*p)

    return d

def f(theta, alpha, beta):
    """        
    Phased F_sim gate.

    Args:
        theta (float)
        alpha (float)
        beta (float)

    Returns:
        (np.array): Phased Fsim gate.
    """
    f = np.zeros(shape=(2,2), dtype=np.complex128)
    cos = np.cos(theta)
    sin = np.sin(theta)
    f[0,0] = cos * np.e**(1j*alpha)
    f[0,1] = sin * np.e**(1j*beta)
    f[1,0] = -sin * np.e**(-1j*beta)
    f[1,1] = cos * np.e**(-1j*alpha)

    return f

def ansatz(nlayers, nmagnons):
    circ = Circuit(2*nmagnons)
    for l in range(2*nmagnons):
        if l%2 != 0:
            circ.add(gates.X(l))
    i = 0
    j = 2*nmagnons-2+1
    for l in range(nlayers):
        q=0
        for _ in range(2*nmagnons-1):
            if q >= i:
                if i == 0:
                    circ.add((gates.GeneralizedfSim(q, q+1, f(0,0,0), 0)))
                elif q < j:
                    circ.add((gates.GeneralizedfSim(q, q+1, f(0,0,0), 0)))

            q=q+1
        i += 2
        j -= 1
    return circ

def loss(params0, nlayers, nmagnons, u, backend):
    circ1 = ansatz(nlayers, nmagnons)
    circ = Circuit(2*nmagnons)
    circ.add(circ1.on_qubits(*reversed(range(2*nmagnons))))
    npars = len(params0)
    params = [(f(params0[i], params0[i+1], params0[i+2]), 0) for i in range(0,npars,3)]
    circ.set_parameters(params)
    state = backend.execute_circuit(circ).state()
    infid = 1 - fidelity(u, state, backend=backend)
    return float(infid)

def print_fun(x, f, accepted):
    if f < 1e-7:
        return True

def get_b_circuit(nqubits, nmagnons, roots, backend=None):
    backend = _check_backend(backend)
    backend.set_precision('double')
    model = XXZ_free_open_model(nqubits, nmagnons)
    model.get_roots(roots)
    model.get_indexes()
    A_N = model.Ark(nmagnons,nqubits)
    C_N = model._get_Crk(nmagnons,nqubits)
    u1 = model.get_u()
    u1 = P_flip(u1)
    u = u1 / np.linalg.norm(u1)
    check = np.max(abs(A_N.T.conj() @ A_N-C_N))
    print(check)

    nlayers = np.max([nmagnons-1,1])
    c1 = ansatz(nlayers, nmagnons)
    c = Circuit(2*nmagnons)
    c.add(c1.on_qubits(*reversed(range(2*nmagnons))))
    
    params0 = np.random.uniform(0,1,(len(c.queue)-nmagnons)*3)
    print('Numerical optimization of the circuit to prepare the initial state')
    result = basinhopping(loss, params0, minimizer_kwargs={"args":(nlayers, nmagnons, u, backend), "method":"L-BFGS-B", 'tol':1e-11} ,disp=True, callback=print_fun)
    params_f = result.x
    npars = len(params_f)
    params = [(f(params_f[i], params_f[i+1], params_f[i+2]), 0) for i in range(0,npars,3)]
    c.set_parameters(params)
    print('\n')
    return c, result, check, u1
