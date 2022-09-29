from fpylll import BKZ as BKZ_FPYLLL
from fpylll import LLL as LLL_FPYLLL
from fpylll import GSO, IntegerMatrix, FPLLL, Enumeration, EnumerationError, EvaluatorStrategy
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic

from keflll_utils import minkowski_embedding, inv_minkowski_embedding, ifft

from copy import deepcopy
import time
RealNumber = RealField(320)
ComplexNumber = ComplexField(320)
RR = RealField(320)
CC = ComplexField(320)

FPLLL.set_precision(1000)

def compare_sage_versions(ver0, ver1):
    #returns -1 if ver0<ver1; 1 if ver0>ver1 and 0 otherwise

    i = ver0.rfind('.')
    j = ver1.rfind('.')

    major0 = int( ver0[:i] )
    major1 = int( ver1[:j] )
    if major0<major1:
        return -1
    if major0>major1:
        return 1

    minor0 = int(ver0[i+1:])
    minor1 = int(ver1[j+1:])

    if minor0<minor1:
        return -1
    if minor0>minor1:
        return 1

    return 0


def randrange_not_null(a,b):
    n = randrange(a,b)
    while n==0:
        n = randrange(a,b)
    return n

def scale_matrix(M):
    #given matrix M over QQ finds least l s.t. l*M is over ZZ. Returns M, k
    if M[0,0].parent()==ZZ:
        return M, 1

    coeffs = [t.denominator() for t in M.coefficients()]

    l = M.denominator()
    return matrix(ZZ,l*M), l

def cvp_for_fractional_lattices(B,target):
    #given basis B over QQ of a lattice and a target vector, returns coordinates of a close vector( wrapper for babai )
    n, m = B.nrows(), B.ncols()

    is_fractional_matrix = False
    l = 1

    if B[0,0].parent() is QQ:
        is_fractional_matrix=True
        B, l = scale_matrix(B)

    B = IntegerMatrix.from_matrix(B)

    t = [float(tmp)*l for tmp in target]
    GSO_M = GSO.Mat(B, float_type='mpfr')
    GSO_M.update_gso()
    c = GSO_M.babai(t)
    return c

def bkz_reduce_fractional_matrix(B, block_size, verbose=False, dump=True):
    n, m = B.nrows(), B.ncols()

    is_fractional_matrix = False
    l = 1
    B_save = deepcopy(B)

    if B[0,0].parent() is QQ:
        is_fractional_matrix=True
        B, l = scale_matrix(B)
    p = max(0, log(max( [abs(t) for t in B.coefficients()]), 2 )-520)
    B *= 2^(-p)
    for i in range(B.nrows()):
        for j in range(B.ncols()):
            B[i,j] = round(B[i,j])

    B = IntegerMatrix.from_matrix(B)

    #BKZ
    GSO_M = GSO.Mat(B, float_type='mpfr')
    GSO_M.update_gso()

    lll_red = LLL_FPYLLL.Reduction(GSO_M)
    lll_red()

    if verbose:
        print('lll slope:', basis_quality(GSO_M)["/"])

    then=time.perf_counter()

    flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS

    then=time.perf_counter()

    block_sizes = [i for i in range(4,int(min(n,m,block_size)),2)]

    gh = gaussian_heuristic([GSO_M.get_r(i,i) for i in range(n)])
    bkz = BKZReduction(GSO_M)
    for beta in block_sizes:    #BKZ reduce the basis
        par = BKZ_FPYLLL.Param(beta,
                               max_loops=12,
                               flags=flags)

        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        #TODO: print in VERBOSE-mode only - done

        if verbose:
            print('bkz for beta=',beta,' done in:', round_time, 'slope:', basis_quality(GSO_M)["/"])
    if verbose:
        print('gh:', log(gh), 'true len:', log(GSO_M.get_r(0,0)))

    dt=time.perf_counter()-then

    print('All BKZ counted in',dt, 'sec')
    B = matrix( GSO_M.B )
    return matrix(B)

def short_lattice_vectors(B, nr_solutions=1,  task_id=None, verbose=verbose ):
    #rerurns Q-coefficients of the (nr_solutions) shortest vectors
    n, m = B.nrows(), B.ncols()

    is_fractional_matrix = False
    l = 1

    if B[0,0].parent() is QQ:
        is_fractional_matrix=True
        B, l = scale_matrix(B)


    B = IntegerMatrix.from_matrix(B)

    #BKZ
    GSO_M = GSO.Mat(B, float_type='mpfr')
    GSO_M.update_gso()

    lll_red = LLL_FPYLLL.Reduction(GSO_M)
    lll_red()

    if verbose:
        print('lll slope:', basis_quality(GSO_M)["/"])

    then=time.perf_counter()

    flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS

    then=time.perf_counter()


    block_sizes = [i for i in range(4,int(min(n,m)),2)]

    gh = gaussian_heuristic([GSO_M.get_r(i,i) for i in range(n)])
    bkz = BKZReduction(GSO_M)
    for beta in block_sizes:    #BKZ reduce the basis
        par = BKZ_FPYLLL.Param(beta,
                               max_loops=14,
                               flags=flags)
        then_round=time.perf_counter()
        bkz(par)
        round_time = time.perf_counter()-then_round
        #TODO: print in VERBOSE-mode only - done

        if verbose:
            print('bkz for beta=',beta,' done in:', round_time, 'slope:', basis_quality(GSO_M)["/"], 'task id =',task_id)
    if verbose:
        print('gh:', log(gh), 'true len:', log(GSO_M.get_r(0,0)))

    dt=time.perf_counter()-then

    print('All BKZ counted in',dt, 'sec. task_id =' + str(task_id))

    r = [GSO_M.get_r(i,i) for i in range(min(n,m))]
    gh = gaussian_heuristic(r)
    R = GSO_M.get_r(0, 0)*1.0

    then = time.perf_counter()
    enum = Enumeration(GSO_M, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS, nr_solutions=int( nr_solutions ) )
    res = enum.enumerate( 0, n, R, 0   )

    print('Enumeration done in', time.perf_counter()-then, ' task id =',task_id)
    if verbose:
        print()
        print('- - -')

    if not is_fractional_matrix:
        return res, B  #if matrix is over ZZ, return coeffs and the basis
    return res, matrix(B)/l    #if over QQ, scale it back and return

def parse_nfhnf_output(output, K):
    ideals_gp = output[1]
    matrix_gp = output[0]

    M = matrix(K,[
        [K(matrix_gp[i][j]) for j in range(len(matrix_gp[0]))] for i in range(len(matrix_gp))
    ])
    Is = [Ideal(K,tmp) for tmp in ideals_gp]
    return M, Is

def nfhnf_pari(A, Is, U=None):
    #A is row pseudomatrix. Is - corresponding ideals
    assert A.nrows() == len(Is)
    K = Is[0].number_field()
    z = K.gen()
    A_ = pari(A)
    I_ = [pari(tmp) for tmp in Is]

    output =  pari.nfhnf( K,[A.transpose(),Is ] )
    M, ideals = parse_nfhnf_output(output,K)

    return M, ideals

def complex_to_list(a):
    re, im = list(a)
    re, im = RR( real(re) ), RR(real(im))
    return [re, im]

class PseudoBasis:

    def __init__(self, A, Is, hnf_reduced=True):
        if len(Is)!=A.nrows():
            raise AssertionError("Num of cols != num of ideals!")

        K = Is[0].number_field()
        self.Field = K
        self.B = None   #lazy QQ embedding

        if hnf_reduced:
            self.A, self.Is = nfhnf_pari(A, Is)
        else:
            self.A, self.Is = A, Is

        self.rank    = A.rank()   #TODO: fix zero ideals issue
        self.degree  = K.degree()
        self.is_hnf_reduced = hnf_reduced

    def __eq__(self, other):

        A = self.A
        B = other.A

        if A.nrows() != B.nrows() or A.ncols() != B.ncols() or self.Field != other.Field:
            return False
        if self.express_as_Q_module() == other.express_as_Q_module():
            return True

        A_hnf = self.hnf_reduce()
        B_hnf = other.hnf_reduce()

        A_hnf.rescale()
        B_hnf.rescale()

        A, I = A_hnf.A, A_hnf.Is
        B, J = B_hnf.A, B_hnf.Is

        if A==B and all( [I[i]==J[i] for i in range(len(I))] ):
            return True

        n, m = A.nrows(), B.ncols()
        U = A.solve_left(B)
        if A_hnf.steinitz_class_member() != det(U) * B_hnf.steinitz_class_member():
            #print('Steinitz contradiction!')
            return False

        for i in range(n):
            for j in range(n):
                if U[i,j] not in I[j]*J[i]^-1:
                    #print('Ideal contradiction!')
                    return False
        return True

    def __contains__(self, key):
        if self.B is None:
            self.B = self.express_as_Q_module()
        K = self.Field
        c = self.A.solve_left(key)

        return all( [ (c[i] in sum( self.Is ) ) for i in range(len(c)) ] )

    def random_element(self, a=None, b=None):
        K = self.Field
        Is = self.Is
        v = vector(K,self.rank)
        if a==None or b==None:
            for i in range(self.rank):
                v += (self.A[i] * K(Is[i].random_element()) )
        else:
            for i in range(self.rank):
                v += (self.A[i] * K(Is[i].random_element()) )
        return v

    def steinitz_class_member(self):
        return prod( self.Is )

    def det(self):
        return norm( det( self.A ) ) * norm( self.steinitz_class_member() )

    def hnf_reduce(self):
        if self.is_hnf_reduced:
            return self

        return( PseudoBasis(self.A,self.Is) )

    def rescale(self, scale='vectors'):
        #Transforms A and Is in a such way that either all A entries are algebraic integers,
        #Or all ideals are integral

        if scale=='vectors':
            for i in range(self.rank):
                tmp=[]
                for j in range(self.rank):
                    tmp += list(self.A[i,j])
                denominators = [t.denominator() for t in tmp]
                l = lcm(denominators)
                self.Is[i]/=l
                self.A[i]*=l
        elif scale == 'ideals':
            for i in range(self.A.nrows()):
                denominator = self.Is[i].norm()
                self.Is[i]/=denominator
                self.A[i]*=denominator


    def express_as_Z_module(self):
        #Works only when all the entries in A are algebraic integers? Otherwise needs to be rescaled.
        M = self.express_as_Q_module()
        M = matrix(ZZ,M)
        return(M)

    def express_as_Q_module(self, do_lll = True):
        #Embeds self to QQ (ZZ if possible).
        A = self.A
        Is = self.Is

        d = self.A[0,0].parent().fraction_field().degree()
        n = A.nrows()
        m = A.ncols()
        beta = [[0 for j in range(d)] for i in range(n)]

        for i in range(n):    #prepeare list of the ideals of the pseudobasis
            b_i_ideal_Z_basis = Is[i].integral_basis()
            for j in range(d):
                beta[i][j] = ( b_i_ideal_Z_basis[j] )

        delta=[[] for t in range(n*d)]
        for i in range(n):   #for n distinct elements to embed
            for j in range(d):    #for d distinct embeddings of each ones
                for x in range(m):
                    tmp = beta[i][j] * A[i,x]   #we embed each component of the vector
                    delta[i*d+j] += list(tmp)
        M = matrix(QQ, delta )

        assert self.det() == det(M), f"Descend to Q failed!" + str(self.det().n(29) )+ ' vs ' + str(det(M).n(29))

        if do_lll:
            M =  M.LLL()
        self.B = M

        return M

    def express_as_Minkowsky_module(self):
        K = self.Field
        z = K.gen()
        d = K.degree()
        sigmas = K.embeddings(CC)[:d// 2 ]
        vectors_alg = [v for v in self.A]

        OK = K.ring_of_integers()

        assert all( [t == Ideal(K,K( 1 )) for t in self.Is] ), f"Minkowski embedding only supported for trivial ideals in pseudobasis"
        A = self.A

        C = []
        for v in A:
            for i in range(d):
                C.append(z**i*v)

        for i in range(len(C)):
            tmp = []
            for j in range(len(C[i])):
                tmp += list( minkowski_embedding(C[i][j]) )
            C[i] = vector(tmp)
        C = matrix(C)
        return C

    def cvp(self,target):
        K=self.Field
        z=K.gen()
        d=K.degree()

        if self.B is None:
            try:
                B = self.express_as_Z_module()
            except:
                B = self.express_as_Q_module()
        else:
            B = self.B

        qtarget=[]
        for tmp in target:
            qtarget += list(tmp)  #coefficient embedding

        s = cvp_for_fractional_lattices(B,qtarget)
        s = vector( [ZZ(tmp) for tmp in s] )*B

        out = []
        for i in range(len(s) / d):
            pairs = s[i*d:(i+1)*d]
            out.append( sum( [z^i * pairs[i] for i in range(d)] ))  #because K(pairs) crashes

        return(vector(out))

    def short_elements(self, nr_solutions= 1 , algebraic_form=False,task_id=None, coords=True, verbose=verbose ):
        #Returns nr_solutions short elements of lattice (coordinates)
        #If algebraic_form == True: vectors returned are the vectors over field self.Field.

        if self.B is None:
            try:
                B = self.express_as_Z_module()
            except:
                B = self.express_as_Q_module()
        else:
            B = self.B

        M, scale = scale_matrix(B)
        if verbose:
            print('max bit length: ', log( max([abs(t) for t in M.coefficients()]), 2).n() )
        coeffs, M =  short_lattice_vectors(M, nr_solutions,  task_id=task_id, verbose=verbose )
        M = matrix(M) / scale

        self.B = M  #TODO: not to do bkz in the next calls

        for i in range(len(coeffs)):
            coeffs[i] = vector(ZZ, [int( round(t) ) for t in coeffs[i][ 1 ]] )   #coeffs consists of floats before this step

        V = [ vector(coeffs[i])*M for i in range(len(coeffs))]

        K = self.Field
        if algebraic_form:
            V_alg = []
            d = self.Field.degree()
            for v in V:
                tmp=list(v)
                vect = []
                for i in range( 0 ,len(v)/d):
                    vect.append( K( list(v[i*d:(i+ 1 )*d]) ) )
                V_alg.append(vector(vect))
            return V_alg

        return V

    def short_elements_via_Minkowski(self, nr_solutions=1 , algebraic_form=False, task_id=None, coords=True, verbose=verbose ):
          #Returns nr_solutions short elements of lattice (coordinates)
          #If algebraic_form == True: vectors returned are the vectors over field self.Field.

          C = self.express_as_Minkowsky_module()    #embed the module into the CC
          B=[]

          for v in C:     #complex to real matrix transformation
              tmp=[]
              for elem in v:
                  tmp += [ RR(real(elem)), RR(imag(elem)) ]   #complex number to pair of RR's (note real(elem) is a CC element!)
              B.append(tmp)

          U = matrix( pari.qflll( matrix(RR, B).transpose() ) ).transpose()  #get LLL transformation...
          B = U*matrix(RR,B)                                                 #and apply it
          TMP = []
          for i in range(B.nrows()):
              TMP.append([])
              for j in range(B.ncols()):
                  TMP[-1].append( ZZ( round( B[i,j] * 2 **300  ) ) * QQ(2) **-300)    #scalem round the matrix and scale it back
          TMP = matrix(QQ,TMP)
          B = TMP

          B , scale = scale_matrix(B)   #scale the matrix to be integral

          coeffs, M =  short_lattice_vectors(B, nr_solutions, task_id=task_id, verbose=verbose )    #find the shortest vectors
          M = matrix(M) / scale
          V = [ vector(coeffs[i][1])*matrix( M ) for i in range(len(coeffs)) ]    #short vector approx coordinates

          M =[]

          self.B = self.express_as_Q_module()   #express self as a Q-module

          for v in V:   #for every short vector

              temp = list( v )
              tmp=[]
              d = self.Field.degree()
              for i in range(len(v) / d):
                  #get tuples of len d and apply inverse minkowski
                  pairs = v[i*d:(i+1)*d]
                  cpairs = [CC(pairs[2*j],pairs[2*j+1]) for j in range(len(pairs)//2)]
                  elem = inv_minkowski_embedding(cpairs)
                  tmp.append(elem)
              ins_cand = []
              for t in tmp:   #due to the rounding error we must force coords to be integral
                  temp = self.Field( [c for c in t] )
                  ins_cand.append(temp)
              ins_cand = self.cvp(ins_cand)
              M.append( vector(ins_cand) )
          return M

    def __str__(self):
        return str( self.A ) + ', ' + str( self.Is )
