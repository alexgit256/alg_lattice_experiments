

# This file was *autogenerated* from the file keflll_utils.sage
from sage.all_cmdline import *   # import sage library

_sage_const_320 = Integer(320); _sage_const_2 = Integer(2); _sage_const_0 = Integer(0); _sage_const_1 = Integer(1); _sage_const_2p = RealNumber('2.'); _sage_const_1p = RealNumber('1.'); _sage_const_4 = Integer(4)
from sys import stdout
import time
import copy
import numpy
from numpy import fft, array
import random as rnd

from fpylll import LLL as LLL_FPYLLL
from fpylll import IntegerMatrix, GSO
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

Prec=_sage_const_320

scale_factor=_sage_const_2 **Prec

ab_situations = {"a0":_sage_const_0 , "0b":_sage_const_0 , "ab":_sage_const_0 , "gcdab":_sage_const_0 }

RealNumber = RealField(Prec)
ComplexNumber = ComplexField(Prec)
RR=RealNumber
CC=ComplexNumber

# - - -


def butterfly(v_,s):
    #butterfly step of fft

    v=[t for t in v_]
    n = len(v_)
    if n>_sage_const_1 :
        vp = v_[_sage_const_0 :n:_sage_const_2 ]
        vi = v_[_sage_const_1 :n:_sage_const_2 ]
        vi = butterfly(vi,s)
        vp = butterfly(vp,s)

        zeta=(exp(-_sage_const_2p *I*pi/n*s)).n(Prec)
        mu=_sage_const_1
        for i in range(n/_sage_const_2 ):
            t=mu*vi[i]
            v_[i+n/_sage_const_2 ]=vp[i]-t
            v_[i]=vp[i]+t
            mu*=zeta
    return v_ if isinstance(v,list) else [v_] #force to return list

def kef_fft(a):
    #KEF version of FFT

    K = a.parent().fraction_field()
    d = K.degree()
    z=K.gen()
    Z=( e**(_sage_const_1p *pi*I/d) ).n(Prec)
    Z=CC(z)

    v = list(a)
    for i in range(len(v)):
        v[i]*=(Z**i)

    v=butterfly(v,_sage_const_1 )
    v = [CC(t) for t in v]

    a_= vector(v)

    return a_

def kef_ifft(v,real_value=True):
    #KEF version of IFFT

    d=len(list(v))
    z=(e**(-_sage_const_1p *pi*I/d)).n(Prec)
    z=CC(z)

    v = list(v)
    v=butterfly(v,-_sage_const_1 )

    for i in range(len(v)):
        v[i]*=(z**i)

    v = [CC(t)/d for t in v] if not real_value else [t[_sage_const_0 ]/d for t in v]

    a_= vector(v)

    return a_

def kef_ifft_numfield(v):
    #vector to the numfield element

    d = len(v)
    assert((d & (d - _sage_const_1 )) == _sage_const_0 )  #check if pow of 2
    K = CyclotomicField(_sage_const_2 *d)

    tmp = kef_ifft(v)
    v_ = [RR(t) for t in tmp]
    v_ = [QQ(t) for t in v_]
    return(K(v_))



def kef_sqrt_numfield(a):
    #checked

    K = a.parent().fraction_field()
    if K.degree() == _sage_const_1 :
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
        a[i]=_sage_const_1 /sqrt(abs(a[i])).n(Prec)
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
    if K.degree()<=_sage_const_4 :
        return a*b

    aa, bb = minkowski_embedding(a), minkowski_embedding(b)
    cc = [aa[i]*bb[i] for i in range(len(a.list())/_sage_const_2 )]
    return inv_minkowski_embedding(cc)

def roundoff(a):
  OK = a.parent().fraction_field().ring_of_integers()
  return OK( [round(t) for t in a] )

def fast_inv(a):
    K = a.parent().fraction_field()
    if K.degree()<=_sage_const_4 :
        return _sage_const_1 /a
    aa = minkowski_embedding(a)
    try:
        aa = [_sage_const_1 /t if t!=_sage_const_0  else _sage_const_0  for t in aa]
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
    if u[_sage_const_0 ].parent().fraction_field().degree()<=_sage_const_4 :
        return u.hermitian_inner_product(v)
    return sum( [fast_mult( v[i], u[i].conjugate() ) for i in range(len(u))] )


def rfact_herm(B, debug=False, Prec=Prec):
    #R factor of RQ decomposition.
    #checked

    RealNumber = RealField(Prec)

    F = B[_sage_const_0 ,_sage_const_0 ].parent()
    d = B.nrows()

    Q_backup = [_sage_const_0  for i in range(d)]   #inner product of gs-vectors buffer

    Q=matrix(F,d,d)
    for j in range(d):
        s = vector(F,d)
        for i in range(j):
            if i == j-_sage_const_1 :
                tmp = sum( fast_mult( Q[i][j], Q[i][j].conjugate() ) for j in range(len(Q[i])) )
                Q_backup[i] = ( fast_inv(tmp) )
            tmp = fast_mult((fast_hermitian_inner_product(Q[i],B[j])) , Q_backup[i])
            s+= vector([fast_mult(tmp,t) for t in Q[i]])
        Q[j]= B[j]-s

    tmp = sum( [fast_mult( Q[d-_sage_const_1 ][j], Q[d-_sage_const_1 ][j].conjugate() ) for j in range(len(Q[d-_sage_const_1 ]))] )
    Q_backup[d-_sage_const_1 ] = (fast_inv(tmp))

    racines = [F(fast_sqrt( Q_backup[j] )) for j in range(d)]

    R=matrix(F,d,d)
    for j in range(d):
        for i in range(j+_sage_const_1 ):
            R[i,j]= fast_mult( fast_hermitian_inner_product(B[j],Q[i]), racines[i]  )  #<b,q>, not <q,b> because hermitian_inner_product works not how we need
    return R


def enorm_numfield(a):
  #returnd squared euclidean norm of a numfield element after coefficient embedding
  tmp = a.list()
  return sum(abs(t)**_sage_const_2  for t in tmp)

def enorm_vector_over_numfield(v):
  return sum( enorm_numfield(t) for t in v )

def ascend(K,v):
    #checked
    #ascends vector to element of the field K

    qh = len(v)
    d_ = v[_sage_const_0 ].parent().degree()
    d=d_*qh
    z_=K.gen()

    v_z = [_sage_const_0 ]*qh*d_

    for i in range(qh):
        for j in range(d_):
            v_z[j*qh+i] = v[i][j]

    out = K(v_z)
    return(out)

def descend(K,a):   #only for K - cyclotomic of power 2
    #checked
    d_ = a.parent().degree()

    out = [_sage_const_0 ,_sage_const_0 ]
    for i in range(_sage_const_2 ):
        out[i] =  K(a.list()[i:d_:_sage_const_2 ])
    return out

def invertibles(f):
    assert f == round(f)
    out=[_sage_const_0  for i in range(euler_phi(f))]

    t=_sage_const_0
    for i in range(f):
        if gcd(i,f)==_sage_const_1 :
            out[t]=i
            t+=_sage_const_1
    return out


def ifft(v,real_value=True):
    #subroutine for inverse minkowsky

    d=len(v)
    z=(e**(-_sage_const_1p *pi*I/d)).n(Prec)
    z=CC(z)

    v = list(v)
    v=butterfly(v,_sage_const_1 )

    for i in range(len(v)):
        v[i]*=(z**i)

    v = [CC(t)/d for t in v] if not real_value else [t[_sage_const_0 ]/d for t in v]

    a_= [QQ(RR(t)) for t in v]
    a_ = [ZZ( round(t*_sage_const_2 **Prec) )*_sage_const_2 **-Prec for t in a_]

    return a_

def minkowski_embedding(a, truncated=True):
    #we have real coefficients so only half the embeddings are enough (truncated=True)
    #applicable only to 2^h cyclotomics where h>=2

    K = a.parent().fraction_field()
    sigmas = K.embeddings(ComplexField(Prec))

    if truncated:
        return vector( [s(a) for s in sigmas[:len(sigmas)/_sage_const_2 ]] )
      #return vector( [s(a) for s in sigmas[0:len(sigmas):2]] )
    else:
        return vector( [s(a) for s in sigmas] )

def inv_minkowski_embedding(s):
    #we have real coefficients so only half the embeddings are enough (truncated=True)
    f = _sage_const_4 *len(s)
    K = CyclotomicField(f, names=('z',)); (z,) = K._first_ngens(1)
    tmp = list( s[:] ) + [_sage_const_0 ]*len(s)
    for i in range(len(s)-_sage_const_1 ,-_sage_const_1 ,-_sage_const_1 ):
        tmp[len(tmp)-_sage_const_1 -i] = s[i].conjugate()

    return K( ifft(tmp) )


def log_embedding(a,truncated=True):
    ac = minkowski_embedding(a,truncated)
    return _sage_const_2 *vector(RealField(Prec), [ln(abs(t).n(Prec)) for t in ac])

def inv_log_embedding(s):
    tmp = [e.n()**(t/_sage_const_2 ) for t in s]
    a = inv_minkowski_embedding(tmp)
    return a


def GEuclide(L, Lptr,a,b):
    #given a, b s.t. (a) is coprime to (b) returns mu, nu s. t. a\mu-b\nu = 1
    g=gcd(norm(a),norm(b))
    try:
        assert g==_sage_const_1 , f"GEuclide: gcd="+str(g)
    except AssertionError as err:
        if g!=_sage_const_0 :
            raise err
        OK = a.parent().fraction_field()
        if a==_sage_const_0  and OK(b).is_unit():
            print('a=0 moment')
            ab_situations["a0"]+=_sage_const_1
            return _sage_const_0 , _sage_const_1 /b
        if b==_sage_const_0  and OK(a).is_unit():
            print('b=0 moment')
            ab_situations["0b"]+=_sage_const_1
            return _sage_const_1 /a, _sage_const_0
        ab_situations["gcdab"]+=_sage_const_1
        raise err
    ab_situations["ab"]+=_sage_const_1
    K = a.parent().fraction_field()
    A = Ideal(K,a)
    B = Ideal(K, b)

    t0, t1 = pari.idealaddtoone(K,A,B)
    t0, t1 = K(t0), K(t1)

    mu, nu = t0/a,  t1/b

    assert a*mu+b*nu in [-_sage_const_1 , _sage_const_1 ]
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

    a,b,c,d = descend(K,B[_sage_const_0 ,_sage_const_0 ]), descend(K,B[_sage_const_0 ,_sage_const_1 ]), descend(K,B[_sage_const_1 ,_sage_const_0 ]), descend(K,B[_sage_const_1 ,_sage_const_1 ])
    #Note: sqrt(z)*[a[0], a[1]] = [z*a[1], a[0]] for sqrt(z) - primitive root of field L (parent field of B)
    #We need such a presentation from [DP16] to ensure that descend_rank2_matrix(A*B) == descend_rank2_matrix(A) * descend_rank2_matrix(B)

    a_= [z*a[_sage_const_1 ], a[_sage_const_0 ]]
    b_= [z*b[_sage_const_1 ], b[_sage_const_0 ]]
    c_= [z*c[_sage_const_1 ], c[_sage_const_0 ]]
    d_= [z*d[_sage_const_1 ], d[_sage_const_0 ]]

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
    a,b = ascend(K,v[_sage_const_0 :_sage_const_2 ]), ascend(K,v[_sage_const_2 :_sage_const_4 ])


    if b==_sage_const_0  and OK(a).is_unit():
        return matrix([
            [_sage_const_0 ,a],
            [_sage_const_1 ,_sage_const_0 ]
        ])

    if a==_sage_const_0  and OK(b).is_unit():
        return matrix([
            [b,_sage_const_0 ],
            [_sage_const_0 ,_sage_const_1 ]
        ])

    try:
        mu, nu = GEuclide(L,Lptr,a,-b)
    except Exception as ex:
    	print('Error:', ex, type(ex))
    	raise ex


    assert abs( norm(a*mu - b*nu) ) == _sage_const_1 , f"Lift: a*mu - b*nu is not a unit!"
    U = matrix(L[Lptr].Field,[
        [a,b],
        [nu,mu]
    ])


    #print('Lift returns norm:',norm(det(U)).n(33))

    assert norm(det(U)) -_sage_const_1   == _sage_const_0 , f"Non-unimodular Lift! "+str(norm(det(U)))+str(a*mu - b*nu)
    #print("lift debug, norm det(U)=", norm(det(U)).n(33))
    return(U)


def compute_log_unit_lattice(K, debug=False):
    #computes log unit lattice for field K
    z_ = K.gen()
    f = z_.multiplicative_order()

    if f<_sage_const_4 :
        return None, None
    else:
        units = [z_**((_sage_const_1 -(_sage_const_2 *i-_sage_const_1 ))/_sage_const_2 ) * (_sage_const_1 -z_**i)/(_sage_const_1 -z_) for i in invertibles(f/_sage_const_2 )[_sage_const_1 :] ]
    assert all( [tmp.is_unit() for tmp in units] )

    d = K.degree()/_sage_const_2
    B=matrix([
        log_embedding(units[i]) * scale_factor for i in range(d-_sage_const_1 )
    ])

    Bint = IntegerMatrix(d-_sage_const_1 ,d)

    for i in range(d-_sage_const_1 ):
        for j in range(d):
            Bint[i,j]=int( B[i,j]  )

    T = IntegerMatrix.identity(d-_sage_const_1 )
    G = GSO.Mat(Bint,float_type='mpfr',U=T)

    lll_ = LLL_FPYLLL.Reduction(G)
    lll_()
    G.update_gso()

    return (G, units)



# - - - for test.sage - - -

def fast_mat_mult(A,B):
    C = matrix(A[_sage_const_0 ,_sage_const_0 ].parent(), A.nrows(), B.ncols())
    for i in range(A.nrows()):
        for j in range(B.ncols()):
            C[i,j] = sum( [fast_mult(A[i,k], B[k,j]) for k in range(A.ncols())] )
    return(C)


class FieldInfo:

    def __init__(self, h):
        self.Field=CyclotomicField(_sage_const_2 **h)

        G, u = compute_log_unit_lattice(self.Field)
        self.LLL_GSO = G
        self.cyclotomic_units = u
