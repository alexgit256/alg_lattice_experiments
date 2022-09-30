from utils import PseudoBasis

def test_qembedding(f=32, n=3, m=4):
    K.<z> = CyclotomicField(f)
    OK = K.ring_of_integers()
    A = matrix(K,[
        [OK.random_element(-10**3, 10**3) for j in range(m)] for i in range(n)
    ])
    I = [ Ideal(K(1)) for i in range(n) ]

    P = PseudoBasis( A, I, hnf_reduced=False )
    B = P.express_as_Q_module()
    #print(P.det()^2)
    #print(norm( det(A*A.conjugate_transpose())).n())

    assert (2*ln( P.det() ) - ln( norm( det(A*A.conjugate_transpose()) ) )) < 0.01, "Modules not equal!"
    print("Sucsess!")
