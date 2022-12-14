from fpylll import BKZ as BKZ_FPYLLL
from fpylll import LLL as LLL_FPYLLL
from fpylll import GSO, IntegerMatrix, FPLLL, Enumeration, EnumerationError, EvaluatorStrategy
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic

import sys

try:
    enorm_vector_over_numfield(vector(CyclotomicField(2),[1]))
except NameError:
    def enorm_numfield(a):
      #returnd squared euclidean norm of a numfield element after coefficient embedding
      tmp = a.list()
      return sum(abs(t)^2 for t in tmp)

    def enorm_vector_over_numfield(v):
      return sum( enorm_numfield(t) for t in v )

from copy import deepcopy
import time
RealNumber = RealField(320)
ComplexNumber = ComplexField(320)
RR = RealField(320)
CC = ComplexField(320)
Prec = 320

FPLLL.set_precision(1000)

def ideal_denominator(I):
    B = [ b.denominator() for b in I.basis() ]
    l = lcm(B)
    return(l)

def roundoff(a):
  #OK = a.parent().fraction_field().ring_of_integers()
  return a.parent()( [round(t) for t in a] )

def butterfly(v_,s):
    #butterfly step of fft

    v=[t for t in v_]
    n = len(v_)
    if n>1:
        vp = v_[0:n:2]
        vi = v_[1:n:2]
        vi = butterfly(vi,s)
        vp = butterfly(vp,s)

        zeta=(exp(-2.*I*pi/n*s)).n(Prec)
        mu=1
        for i in range(n/2):
            t=mu*vi[i]
            v_[i+n/2]=vp[i]-t
            v_[i]=vp[i]+t
            mu*=zeta
    return v_ if isinstance(v,list) else [v_] #force to return list

def ifft(v,real_value=True):
    #subroutine for inverse minkowsky

    d=len(v)
    z=(e**(-1.*pi*I/d)).n(Prec)
    z=CC(z)

    v = list(v)
    v=butterfly(v,1)

    for i in range(len(v)):
        v[i]*=(z^i)

    v = [CC(t)/d for t in v] if not real_value else [t[0]/d for t in v]

    a_= [QQ(RR(t)) for t in v]

    return a_

def minkowski_embedding(a, truncated=True):
    #we have real coefficients so only half the embeddings are enough (truncated=True)
    #applicable only to 2^h cyclotomics where h>=2

    K = a.parent().fraction_field()
    sigmas = K.embeddings(ComplexField(Prec))

    if truncated:
        return [s(a) for s in sigmas[:len(sigmas)/2]]
    else:
        return [s(a) for s in sigmas]

def inv_minkowski_embedding(s):
    #we have real coefficients so only half the embeddings are enough (truncated=True)
    f = 4*len(s)
    K.<z> = CyclotomicField(f)
    tmp = list( s[:] ) + [0]*len(s)
    for i in range(len(s)-1,-1,-1):
        tmp[len(tmp)-1-i] = s[i].conjugate()

    return K( ifft(tmp) )

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
    n = 0
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
            print('bkz for beta=',beta,' done in:', round_time, 'slope:', basis_quality(GSO_M)["/"], 'task_id = ', task_id)
    if verbose:
        print('gh:', log(gh), 'true len:', log(GSO_M.get_r(0,0)))

    dt=time.perf_counter()-then

    print('All BKZ counted in',dt, 'sec','task_id = ', task_id)
    B = matrix( GSO_M.B )
    return matrix(B)

def short_lattice_vectors(B, nr_solutions=1,  task_id=None, verbose=verbose, approx=None, radius=0.995 ):
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

    if approx is None:
        block_sizes = [i for i in range(4,int(min(n,m)),2)]
    else:
        assert approx in ZZ and approx > 0, f"approx sould be non-negative integer!"
        block_sizes = [i for i in range(4,int(min(n,m,approx)),2)] + [ min(n,m,approx) ]

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
            sys.stdout.flush()
    if verbose:
        GSO_M.update_gso()
        gh = gaussian_heuristic([GSO_M.get_r(i,i) for i in range(n)])
        print('gh:', gh^0.5 / l, 'true len:', GSO_M.get_r(0,0)^0.5 / l )

    if not approx is None:
        B = matrix(B)
        res =   [ (None, B.solve_left( vector(B[i]) )) for i in range(nr_solutions)]
        return res, B/l

    dt=time.perf_counter()-then

    print('All BKZ counted in',dt, 'sec')

    r = [GSO_M.get_r(i,i) for i in range(min(n,m))]
    gh = gaussian_heuristic(r)
    R = GSO_M.get_r(0, 0)*radius

    then = time.perf_counter()
    enum = Enumeration(GSO_M, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS, nr_solutions=int( nr_solutions ) )
    res = enum.enumerate( 0, n, R, 0   )

    print('Enumeration done in', time.perf_counter()-then, 'task_id = ', task_id)
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
    assert A.nrows() == len(Is), "Wrond dimensions in nfhnf_pari"
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

    def __init__(self, A, Is, hnf_reduced=False):
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
        K = self.Field
        c = self.A.solve_left(key)

        return all( [ (c[i] in self.Is[i] ) for i in range(len(c)) ] )

    def bitlength(self):
        try:
            B = self.express_as_Z_module()
        except:
            B = self.express_as_Q_module()
        Tmp = self.B
        tmp = max( Tmp.coefficients() )

        return ( log(tmp,2) + log(Tmp.denominator(),2) ).n()

    def random_element(self, a=None, b=None):
        K = self.Field
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
        if self.A.nrows() == self.A.ncols():
            return norm( det( self.A ) ) * norm( self.steinitz_class_member() )
        return norm( det( self.A * self.A.conjugate_transpose() ) )^(1/2) * norm( self.steinitz_class_member() )

    def hnf_reduce(self):
        if self.is_hnf_reduced:
            return self

        return( PseudoBasis(self.A,self.Is, hnf_reduced=True) )

    def rescale(self, scale='vectors'):
        #Transforms A and Is in a such way that either all A entries are algebraic integers,
        #Or all ideals are integral
        self.is_hnf_reduced = False
        if scale=='vectors':
            for i in range(self.rank):
                tmp=[]
                for j in range(self.rank):
                    tmp += list(self.A[i,j])
                denominators = [t.denominator() for t in tmp]
                l = lcm(denominators)
                self.Is[i]/=l
                self.A[i]*=l
        elif scale == 'integral_ideals':
            for i in range(self.A.nrows()):
                denominator = lcm( [elem.denominator() for elem in self.Is[i].basis()] )  #find a s.t. a*Is[i] is integral
                self.Is[i]*=denominator
                self.A[i]/=denominator
        elif scale == 'ring_of_integers':
            K = self.Field
            for i in range(self.A.nrows()):
                a, b = pari.bnfisprincipal( pari.bnfinit(K), self.Is[i] )
                if a == [] and b == []:
                    continue
                if a == 0 or a == []:
                    b = K(b)
                    self.A[i]*=b
                    self.Is[i] /= b

    def make_almost_free( self ):
        #Proposition 1.3.6 and 1.3.12 from Cohen

        self.is_hnf_reduced = False
        self.rescale('integral_ideals')
        A = self.A
        Is = self.Is
        K = self.Field
        for i in range( A.nrows()-1 ):    # for every submodule A*x+B*y -> OK*x' + AB*y'
            aid, bid = Is[i], Is[i+1]

            sa, sb = ideal_denominator(aid), ideal_denominator(bid)   #scale coefficients

            aid, bid = sa*Is[i], sb*Is[i+1]     #scale the ideals
            A[i], A[i+1] = A[i]/sa, A[i+1]/sb   #dont' forget to scale the vectors

            #corollary 1.2.11 from Cohen
            aid_m1 = aid^-1
            scale = ideal_denominator(aid_m1)
            aid_m1 *= scale
            if log(scale,2).n() > 350:    #if "PariError: the PARI stack overflows" is about to happen...
                pari.default("parisizemax", 2**33)
                pari.allocatemem(2**33)
            a = pari.idealcoprime(K,aid_m1,bid) #  a*(scale*aid^-1) is coprime to bid and is integral

            a = K(a)*scale # a*aid^-1 is coprime to bid and is integral
            e, f = pari.idealaddtoone(K,a*aid^-1,bid)
            e, f = K(e), K(f)
            assert f in bid
            assert e+f == K(1)
            b, c = f, -1
            d = e/a

            A[i], A[i+1] = A[i]*a + A[i+1]*b, A[i]*c + A[i+1]*d
            Is[i], Is[i+1] = Ideal(K,K(1)), Is[i]*Is[i+1]
        self.A = A
        self.Is = Is
        self.rescale('ring_of_integers')
        self.B = self.express_as_Q_module()

    def express_as_Z_module(self, do_lll = True):
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

        #The assertion below fails because of https://trac.sagemath.org/ticket/34597. That's why we need file test_Qembedding.sage
        #assert abs(1 - self.det() / det(M*M.conjugate_transpose())^(1/2))<0.05, f"Descend to Q failed!" + str(self.det().n(29) )+ ' vs ' + str(det(M*M.transpose())^0.5.n(29))

        if do_lll:
            M =  M.LLL()
        self.B = M

        return M

    def express_as_Minkowsky_module(self):
        K = self.Field
        z = K.gen()
        d = K.degree()
        sigmas = K.embeddings(CC)[:d//2 ]
        vectors_alg = [v for v in self.A]

        OK = K.ring_of_integers()

        assert all( [t == Ideal(K,K(1 )) for t in self.Is] ), f"Minkowski embedding only supported for trivial ideals in pseudobasis"
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
        #givet the target (in algebraic form) returns the result of the Babai nearest plane algorithm
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

    def gaussian_heuristic(self):
        if self.B is None:
            try:
                B = self.express_as_Z_module()
                l=1
            except:
                B = self.express_as_Q_module()
                B,l = scale_matrix(B)
        else:
            B = self.B
            B,l = scale_matrix(B)

        Bint = IntegerMatrix.from_matrix(B)
        g = GSO.Mat(Bint)
        g.update_gso()
        gh = gaussian_heuristic([g.get_r(i,i) for i in range(Bint.nrows)])
        return (gh )^(1/2) / l

    def minkowski_bound(self):
        dim = self.rank * self.Field.degree()
        return sqrt(dim) * (self.det())^(1/dim)

    def short_elements(self, nr_solutions=1 , algebraic_form=True,task_id=None, coords=True, verbose=verbose, approx=None, radius=1.03 ):
        #Returns nr_solutions short elements of lattice (coordinates)
        #If algebraic_form == True: vectors returned are the vectors over field self.Field.
        #Else return vectors over QQ

        if self.B is None:
            try:
                B = self.express_as_Z_module()
            except:
                B = self.express_as_Q_module()
        else:
            B = self.B

        #M, scale = scale_matrix(B)
        M = B
        if verbose:
            print('max bit length: ', log( max([abs(t) for t in M.coefficients()]), 2).n() )
        coeffs, M =  short_lattice_vectors(M, nr_solutions,  task_id=task_id, verbose=verbose, approx=approx, radius=radius )
        M = matrix(M)


        self.B = M  #TODO: not to do bkz in the next calls

        for i in range(len(coeffs)):
            coeffs[i] = vector(ZZ, [int( round(t) ) for t in coeffs[i][1 ]] )   #coeffs consists of floats before this step

        V = [ vector(coeffs[i])*M for i in range(len(coeffs))]

        K = self.Field
        if algebraic_form:
            V_alg = []
            d = self.Field.degree()
            for v in V:
                tmp=list(v)
                vect = []
                for i in range(0 ,len(v)/d):
                    vect.append( K( list(v[i*d:(i+1 )*d]) ) )
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
