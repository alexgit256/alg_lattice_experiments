import time
import copy
import numpy
from numpy import fft, array
import random as rnd

from fpylll import LLL as LLL_FPYLLL
from fpylll import IntegerMatrix, GSO
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

Prec=320

scale_factor=2^Prec

ab_situations = {"a0":0, "0b":0, "ab":0, "gcdab":0}

RealNumber = RealField(Prec)
ComplexNumber = ComplexField(Prec)
RR=RealNumber
CC=ComplexNumber

# - - -


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

def kef_fft(a):
    #KEF version of FFT

    K = a.parent().fraction_field()
    d = K.degree()
    z=K.gen()
    Z=( e^(1.*pi*I/d) ).n(Prec)
    Z=CC(z)

    v = list(a)
    for i in range(len(v)):
        v[i]*=(Z**i)

    v=butterfly(v,1)
    v = [CC(t) for t in v]

    a_= vector(v)

    return a_

def kef_ifft(v,real_value=True):
    #KEF version of IFFT

    d=len(list(v))
    z=(e**(-1.*pi*I/d)).n(Prec)
    z=CC(z)

    v = list(v)
    v=butterfly(v,-1)

    for i in range(len(v)):
        v[i]*=(z^i)

    v = [CC(t)/d for t in v] if not real_value else [t[0]/d for t in v]

    a_= vector(v)

    return a_

def kef_ifft_numfield(v):
    #vector to the numfield element

    d = len(v)
    assert((d & (d - 1)) == 0)  #check if pow of 2
    K = CyclotomicField(2*d)

    tmp = kef_ifft(v)
    v_ = [RR(t) for t in tmp]
    v_ = [QQ(t) for t in v_]
    return(K(v_))



def kef_sqrt_numfield(a):
    #checked

    K = a.parent().fraction_field()
    if K.degree() == 1:
        return sqrt(abs(a))

    a = kef_fft(a)
    for i in range(len(a)):
        a[i]=sqrt((a[i].n(Prec)))
    tmp = [QQ(t) for t in kef_ifft(a)]
    return K([QQ(t) for t in tmp])

def kef_inv_sqrt_numfield(a):
    #checked

    # returns b such that b*conj(b) = a
    K = a.parent().fraction_field()
#     if K.degree()<=4:
#         return 1/a
    a = kef_fft(a)
    for i in range(len(a)):
        a[i]=1/sqrt(abs(a[i])).n(Prec)
    tmp = [QQ(t) for t in kef_ifft(a)]
    return K([QQ(t) for t in tmp])

def fast_mult(a,b):
  #checked
    K, L = a.parent().fraction_field(), b.parent().fraction_field()
    if not K==L:
        if K.is_subring(L):
            a=L(a)
        elif L.is_subring(K):
            b=K(b)
        else:
            print("Numfield incompatible!")
    if K.degree()<=4:
        return a*b

    aa, bb = minkowski_embedding(a), minkowski_embedding(b)
    cc = [aa[i]*bb[i] for i in range(len(a.list())/2)]
    return inv_minkowski_embedding(cc)

def roundoff(a):
  OK = a.parent().fraction_field().ring_of_integers()
  return OK( [round(t) for t in a] )

def fast_inv(a):
    K = a.parent().fraction_field()
    if K.degree()<=4:
        return 1/a
    aa = minkowski_embedding(a)
    try:
        aa = [1/t if t!=0 else 0 for t in aa]
    except ZeroDivisionError:
        raise ZeroDivisionError("FFT inverse doesn't exist!")

    return inv_minkowski_embedding(aa)

def fast_sqrt(a):
    if a in RR or a in QQ:
        return QQ( sqrt(abs(a.n(Prec)) ))
    tmp = minkowski_embedding(a)

    tmp = [sqrt(t.n(Prec)) for t in tmp]

    return inv_minkowski_embedding(tmp)

def fast_hermitian_inner_product(u,v):
    if u[0].parent().fraction_field().degree()<=4:
        return u.hermitian_inner_product(v)
    return sum( [fast_mult( v[i], u[i].conjugate() ) for i in range(len(u))] )


def rfact_herm(B, debug=False, Prec=Prec):
    #R factor of RQ decomposition.
    #checked

    RealNumber = RealField(Prec)

    F = B[0,0].parent()
    d = B.nrows()

    Q_backup = [0 for i in range(d)]   #inner product of gs-vectors buffer

    Q=matrix(F,d,d)
    for j in range(d):
        s = vector(F,d)
        for i in range(j):
            if i == j-1:
                tmp = sum( fast_mult( Q[i][j], Q[i][j].conjugate() ) for j in range(len(Q[i])) )
                Q_backup[i] = ( fast_inv(tmp) )
            tmp = fast_mult((fast_hermitian_inner_product(Q[i],B[j])) , Q_backup[i])
            s+= vector([fast_mult(tmp,t) for t in Q[i]])
        Q[j]= B[j]-s

    tmp = sum( [fast_mult( Q[d-1][j], Q[d-1][j].conjugate() ) for j in range(len(Q[d-1]))] )
    Q_backup[d-1] = (fast_inv(tmp))

    racines = [F(fast_sqrt( Q_backup[j] )) for j in range(d)]

    R=matrix(F,d,d)
    for j in range(d):
        for i in range(j+1):
            R[i,j]= fast_mult( fast_hermitian_inner_product(B[j],Q[i]), racines[i]  )  #<b,q>, not <q,b> because hermitian_inner_product works not how we need
    return R


def enorm_numfield(a):
  #returnd squared euclidean norm of a numfield element after coefficient embedding
  tmp = a.list()
  return sum(abs(t)^2 for t in tmp)

def enorm_vector_over_numfield(v):
  return sum( enorm_numfield(t) for t in v )

def ascend(K,v):
    #checked
    #ascends vector to element of the field K

    qh = len(v)
    d_ = v[0].parent().degree()
    d=d_*qh
    z_=K.gen()

    v_z = [0]*qh*d_

    for i in range(qh):
        for j in range(d_):
            v_z[j*qh+i] = v[i][j]

    out = K(v_z)
    return(out)

def descend(K,a):   #only for K - cyclotomic of power 2
    #checked
    d_ = a.parent().degree()

    out = [0,0]
    for i in range(2):
        out[i] =  K(a.list()[i:d_:2])
    return out

def invertibles(f):
    assert f == round(f)
    out=[0 for i in range(euler_phi(f))]

    t=0
    for i in range(f):
        if gcd(i,f)==1:
            out[t]=i
            t+=1
    return out


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
    a_ = [ZZ( round(t*2**Prec) )*2**-Prec for t in a_]

    return a_

def minkowski_embedding(a, truncated=True):
    #we have real coefficients so only half the embeddings are enough (truncated=True)
    #applicable only to 2^h cyclotomics where h>=2

    K = a.parent().fraction_field()
    sigmas = K.embeddings(ComplexField(Prec))

    if truncated:
        return vector( [s(a) for s in sigmas[:len(sigmas)/2]] )
      #return vector( [s(a) for s in sigmas[0:len(sigmas):2]] )
    else:
        return vector( [s(a) for s in sigmas] )

def inv_minkowski_embedding(s):
    #we have real coefficients so only half the embeddings are enough (truncated=True)
    f = 4*len(s)
    K.<z> = CyclotomicField(f)
    tmp = list( s[:] ) + [0]*len(s)
    for i in range(len(s)-1,-1,-1):
        tmp[len(tmp)-1-i] = s[i].conjugate()

    return K( ifft(tmp) )


def log_embedding(a,truncated=True):
    ac = minkowski_embedding(a,truncated)
    return 2*vector(RealField(Prec), [ln(abs(t).n(Prec)) for t in ac])

def inv_log_embedding(s):
    tmp = [e.n()^(t/2) for t in s]
    a = inv_minkowski_embedding(tmp)
    return a


def GEuclide(L, Lptr,a,b):
    #given a, b s.t. (a) is coprime to (b) returns mu, nu s. t. a\mu-b\nu = 1
    g=gcd(norm(a),norm(b))
    try:
        assert g==1, f"GEuclide: gcd="+str(g)
    except AssertionError as err:
        if g!=0:
            raise err
        OK = a.parent().fraction_field()
        if a==0 and OK(b).is_unit():
            print('a=0 moment')
            ab_situations["a0"]+=1
            return 0, 1/b
        if b==0 and OK(a).is_unit():
            print('b=0 moment')
            ab_situations["0b"]+=1
            return 1/a, 0
        ab_situations["gcdab"]+=1
        raise err
    ab_situations["ab"]+=1
    K = a.parent().fraction_field()
    A = Ideal(K,a)
    B = Ideal(K, b)

    t0, t1 = pari.idealaddtoone(K,A,B)
    t0, t1 = K(t0), K(t1)

    mu, nu = t0/a,  t1/b

    assert a*mu+b*nu in [-1, 1]
    assert mu in OK and nu in OK, f"idealaddtoone didn't return solutions from the OK!"

    """
    Here we do the size reduction
    W = matrix(K,[
        [a, b],
        [nu_,mu_]
    ])
    R = rfact_herm(W)
    V = size_reduce(R, W, L[Lptr])   #this changes W
    #   Retrieving parasiting unit. unnessesary for Lemma 4.
    nu, mu = V[1]*W
    assert a*mu+b*nu in [-1, 1]
    """

    return mu, nu

def descend_rank2_matrix(K,B):
    #Descend matrix to the subfield K.
    z=K.gen()

    a,b,c,d = descend(K,B[0,0]), descend(K,B[0,1]), descend(K,B[1,0]), descend(K,B[1,1])
    #Note: sqrt(z)*[a[0], a[1]] = [z*a[1], a[0]] for sqrt(z) - primitive root of field L (parent field of B)
    #We need such a presentation from [DP16] to ensure that descend_rank2_matrix(A*B) == descend_rank2_matrix(A) * descend_rank2_matrix(B)

    a_= [z*a[1], a[0]]
    b_= [z*b[1], b[0]]
    c_= [z*c[1], c[0]]
    d_= [z*d[1], d[0]]

    C = matrix(K,[
        a  + b,
        a_ + b_,
        c  + d,
        c_ + d_
    ])
    return C


def Lift(L,Lptr,v,debug=False):
  # Lemma 4 condition checked
    K=L[Lptr].Field
    OK=K.ring_of_integers()
    a,b = ascend(K,v[0:2]), ascend(K,v[2:4])


    if b==0 and OK(a).is_unit():
        return matrix([
            [0,a],
            [1,0]
        ])

    if a==0 and OK(b).is_unit():
        return matrix([
            [b,0],
            [0,1]
        ])

    try:
        mu, nu = GEuclide(L,Lptr,a,-b)
    except Exception as ex:
    	print('Error:', ex, type(ex))
    	raise ex


    assert abs( norm(a*mu - b*nu) ) == 1, f"Lift: a*mu - b*nu is not a unit!"
    U = matrix(L[Lptr].Field,[
        [a,b],
        [nu,mu]
    ])


    #print('Lift returns norm:',norm(det(U)).n(33))

    assert norm(det(U)) -1  == 0, f"Non-unimodular Lift! "+str(norm(det(U)))+str(a*mu - b*nu)
    #print("lift debug, norm det(U)=", norm(det(U)).n(33))
    return(U)


def compute_log_unit_lattice(K, debug=False):
    #computes log unit lattice for field K
    z_ = K.gen()
    f = z_.multiplicative_order()

    if f<4:
        return None, None
    else:
        units = [z_^((1-(2*i-1))/2) * (1-z_^i)/(1-z_) for i in invertibles(f/2)[1:] ]
    assert all( [tmp.is_unit() for tmp in units] )

    d = K.degree()/2
    B=matrix([
        log_embedding(units[i]) * scale_factor for i in range(d-1)
    ])

    Bint = IntegerMatrix(d-1,d)

    for i in range(d-1):
        for j in range(d):
            Bint[i,j]=int( B[i,j]  )

    T = IntegerMatrix.identity(d-1)
    G = GSO.Mat(Bint,float_type='mpfr',U=T)

    lll_ = LLL_FPYLLL.Reduction(G)
    lll_()
    G.update_gso()

    return (G, units)



# - - - for test.sage - - -

def fast_mat_mult(A,B):
    C = matrix(A[0,0].parent(), A.nrows(), B.ncols())
    for i in range(A.nrows()):
        for j in range(B.ncols()):
            C[i,j] = sum( [fast_mult(A[i,k], B[k,j]) for k in range(A.ncols())] )
    return(C)


class FieldInfo:

    def __init__(self, h):
        self.Field=CyclotomicField(2^h)

        G, u = compute_log_unit_lattice(self.Field)
        self.LLL_GSO = G
        self.cyclotomic_units = u
