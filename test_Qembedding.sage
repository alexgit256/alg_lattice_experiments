def express_as_Q_module(A, Is):
    d = A[0,0].parent().degree()
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
    M = matrix(ZZ, delta )

    #assert self.det() == det(M), f"Descend to Q failed!" + str(self.det().n(29) )+ ' vs ' + str(det(M).n(29))

    return M

K.<z> = CyclotomicField(2^4)
OK = K.ring_of_integers()
a = OK([randrange(1, 10^3) for i in range(K.degree())])
print('a:', a)
A = matrix([[a]])


res = express_as_Q_module(A, [Ideal(K(1))])
res2 = matrix([
    list(tmp) for tmp in Ideal(a).basis()
])

print(res)
print(res2)
U = (res2*res^-1)
print(U)
print(det(U))

res = matrix(ZZ,res)
res2 = matrix(ZZ,res2)

Zero = res.hermite_form() - res2.hermite_form()

print( Zero )

assert Zero.is_zero()
