from utils import PseudoBasis, randrange_not_null, compare_sage_versions, scale_matrix
from keflll_utils import FieldInfo, descend_rank2_matrix, enorm_vector_over_numfield, ascend, enorm_numfield
from keflll_utils import rfact_herm, invertibles
from multiprocessing import Pool, cpu_count
from sage import version
import random as rnd
from time import perf_counter

from fpylll import IntegerMatrix, GSO
from fpylll.tools.quality import basis_quality

from sage.misc.randstate import current_randstate

import multiprocessing
import sage.parallel.multiprocessing_sage


def rand_unit(K, c=11):
    z = K.gen()
    OK = K.ring_of_integers()
    U = [OK( (1-z^(2*i+1))/(1-z) ) for i in range(K.degree()/2) ]
    u = 1
    for t in U:
        u*=t**randrange(-c,c+1)
    return u

def gen_unimod_matrix(K,n, rng=[-3,3], density=0.99):
    z=K.gen()
    OK = K.fraction_field().ring_of_integers()

    units = [z^((1-(i))/2) * (1-z^i)/(1-z) for i in invertibles(K.degree()) ] if K.degree()>2 else [1,z]

    L = matrix.identity(K,n)
    U = matrix.identity(K,n)
    for i in range(n):
        for j in range(n):
            if i<j:
                if uniform(0,1)<density:
                    L[i,j]=OK.random_element(rng[0],rng[1])
            elif j<i:
                if uniform(0,1)<density:
                    U[n-j-1,n-i-1]=OK.random_element(rng[0],rng[1])
            else:
                U[i,i]*=prod([ units[randrange(len(units))] for i in range(6) ])
                L[i,j]*=prod([ units[randrange(len(units))] for i in range(6) ])

    return L*U

def add_to_one(a,b):
    #Returns mu, nu s.t. a*mu+b*nu is unit or raises an exception.
    K = a.parent().fraction_field()
    OK = K.ring_of_integers()
    I = Ideal(K,a)
    J = Ideal(K,b)

    mu, nu =  pari.idealaddtoone(K,I,J)
    mu, nu = K(mu), K(nu)
    if a!=0:
        mu = mu / a
    if b!=0:
        nu = nu / b

    if (a*mu + b*nu).is_unit():
        return K(mu), K(nu)
    if (b*mu + a*nu).is_unit():
        return K(nu), K(mu)       #if pari swaps the answers

def test_short_vector_insertion(h,c=10**3,task_id=None, verbose=False, via_Minkowski=False, check_module_equality=False, do_rfact=False, qary=None):
    """
    Builds rank-2 module over h th cyclotomic field. Finds the heuristically
    shortest vector and attempts to insert it into basis. Logs whether it
    manages to or fails.
    --------------------------------------------------------------------------
    h: field conductor
    c: range of initial coefficients
    task_id: used for enumerating experiments
    via_Minkowski: if True use Minkowski embedding
    check_module_equality: check that we obtained the equivalent module by comparing their hnf forms. Slow for h>=6.
    """


    if euler_phi(h)>36:
        print("Warning! euler_phi(h)>36: programm might work extremely slow!")

    K.<z> = CyclotomicField(h)

    OK = K.ring_of_integers()

    text_output = ''

    """
    B is the matrix of the random module. It is aslo the pseudomatrix of pseudobasis
    of the same module with ideals that are OK.
    """
    set_random_seed(hash(task_id)+int(round(2**64*perf_counter())))   #rerandomize, because this is ran in parallel
    randrange = current_randstate().python_random().randrange
    lll_slope = Infinity
    detB = 0
    while lll_slope > 0 or detB==0:  #resample modules untill the slope is negative
        if lll_slope<Infinity and verbose:
            print("Slope>0. Resample!")

        if qary is None:
            b00, b01 = [ OK([randrange(c,c*10+1)*randrange_not_null(-1,2) for t in range(K.degree()) ]) for i in range(2) ]
            b10, b11 = [ OK([randrange(c,c*10+1)*randrange_not_null(-1,2) for t in range(K.degree()) ]) for i in range(2) ]
            B = matrix(K, [
              [ b00*rand_unit(K) ,  b01*rand_unit(K)],
              [ b10*rand_unit(K) ,  b11*rand_unit(K)]
            ])
        else:
            b10 = OK([randrange(0,qary) for i in range(K.degree())])
            B = matrix(K, [
              [ K(qary) ,  0],
              [ b10 ,  K(1)]
            ])

        if do_rfact and (h & (h-1) == 0) and n != 0:    #if we try to do rfact on pow-of-2 cyclotomics...
            B = rfact_herm(B)   #we do so
            assert B[0,1]==0, f"error in rfact_herm output"
        elif do_rfact and (h & (h-1) != 0):
            raise AssertionError("Rfact supported only for pow-of-2 cyclotomics!")

        detB = det(B)
        if detB==0:   #if rank<2 then resample
            continue

        #an object to solve SVP for the module:
        pseudomatrix = PseudoBasis(B, [Ideal(K,K(1)) for i in range(B.nrows())], hnf_reduced=False)  #construct the pseudobasis object
        M = pseudomatrix.express_as_Q_module()
        M, l = scale_matrix( M )
        M = IntegerMatrix.from_matrix(M)
        g = GSO.Mat(M)
        g.update_gso()
        lll_slope = basis_quality(g)["/"]

    #note! this one returns coords!
    if not via_Minkowski:
        v = pseudomatrix.short_elements(nr_solutions=1, algebraic_form=True,task_id=task_id, verbose=verbose)[0]
    else:
        v = pseudomatrix.short_elements_via_Minkowski(nr_solutions=1, algebraic_form=True,task_id=task_id, verbose=verbose)[0]

    #This one is sanity check.
    assert v in pseudomatrix, f"Found vector not in module!"

    #text_output += str("Norm of the short vector: {} \n".format(enorm_vector_over_numfield(v).n()))

    ab = B.solve_left(v)

    #text_output += 'Short vectors coords: \n'
    #text_output += 'Coordinates:' + str(ab) + '\n'

    #assert all( [t in J.ring_of_integers() for t in ab] ), f"Fatal error! Non-integral coordinates!"

    s0 = v[0]
    s1 = v[1]

    #text_output += 's0={}\n'.format(s0)
    #text_output += 's1={}\n'.format(s1)

    if verbose:
        print(text_output)

    """
    The pseudomatrix of the module's pseudobasis is:
    | b0   0 |  OK
    | c0  c1 |  OK

    This is implemented here by storing just the matrix (we know the ideals are OK).
    This code searches for the shortest vector of the module and attempts to insert it
    into the basis using Lift(). It then writes down one of the 8 possible situations that
    can occur.
    """

    #We have next ideals as the submodules of the original module generated by B_up:
    b0O  = Ideal(K, B[0,0] )                       #b0 * OK

    """
    We have 8 endstates of the experiment:
    0: v is in the form (s0,0) and (s0)!=(b0)
    1: v is in the form (s0,0) and (s1)==(b0)
    2: v is in the form (s0,s1)     #just for curiosity
    3: v is in the form (0,s1)      #just for curiosity
    Note  : difference between 4 and 6 is that whether (s0) is the sum of (b0) and (c0)
    Note 2: difference between 5 and 7 is the same

    If s1=0, we know that vector (c0, c1) cannot be in linerar combination for (s0, 0),
    so we have 2 options here (no. 0 and 1)
    """

    #Checks which one of the (s0,0), (0,s1), (s0,s1) the situation is:
    if s0==0:
        vect_shape = '0s'
    elif s1==0:
        vect_shape = 's0'
    else:
        vect_shape = 'ss'
    s0_short_condition = Ideal(K,s0) != b0O

    if vect_shape == 's0':
        if s0_short_condition:
            experiment_context_flag = 0
        else:
            experiment_context_flag = 1
    elif vect_shape == 's0':
        experiment_context_flag = 2
    else:
        experiment_context_flag = 3
    if verbose:
        print(f"experiment_context_flag = {experiment_context_flag} at id= {task_id}")

    try:
        mu, nu = add_to_one(ab[0],ab[1])
        nu = -nu  #we are searching for the solutions of a*mu-b*nu = 1
        OK(mu), OK(nu)  #if not integral, we\'ll get error

        assert (ab[0]*mu-ab[1]*nu).is_unit(), f"Not a Bezout solution! " + str(ab[0]*mu+ab[1]*nu) + str((mu,nu))
        U = matrix(K,[
        [ab[0],ab[1]],
        [nu, mu]
        ])
        if verbose:
            print("Lift sucsess!")

    except Exception as err:
        if verbose:
            print(err)
        return experiment_context_flag, False

    M = U*B
    """
    if verbose:
        print('M lengths:')
        for b in M:
            print(enorm_vector_over_numfield(b).n())
    """

    pseudomatrix_post = PseudoBasis(M, [Ideal(K(1)) for i in range(M.nrows())], hnf_reduced=False)

    if check_module_equality:
        try:
            assert pseudomatrix_post.hnf_reduce()==pseudomatrix.hnf_reduce(), f"Modules are not equal!"
        except AssertionError as err:
            if verbose:
                print(err)
                print(U)
                print(pseudomatrix_post.det().n(),pseudomatrix.det().n() )
            return experiment_context_flag, False
    return experiment_context_flag, True

def test_short_vector_insertion_multiple_times(
    h,                #conductor of the field
    times=5,          #number of the experiments
    c=10**7,          #magnitude of coefficients
    verbose = False,
    via_Minkowski = False,  #use Minkowski embedding
    check_module_equality = False,  #check if the module after the insertion is equal to the original one. Relatively slow and always should pass if no bugs.
    do_rfact = False,        #do rfact_herm on the matrix to simulate KEF-LLL routin
    nthreads = 1,           #amount of threads to utilize (<2 if not using multithreading at all)
    qary = None             #if not None, should be an integer q. Then the module is (q,0)*OK + ([h mod q],1)*OK
    ):

    succ = 0
    state_stat = [[0,0] for i in range(4)]
    print('Lattice dimension: ', 2*euler_phi(h))
    then = perf_counter()
    if compare_sage_versions(version.version, '9.5') <1 and nthreads>1:
        proc = min( max(1,cpu_count()-2), nthreads )
        pool = Pool(processes = proc )
        print(f"Launching experiments on %d threads..." % proc)
        print('______________________________________')
        outputs = []
        tasks = []
        for i in range(times):
            task_id = 'id_'+str(i)
            tasks.append(  pool.apply_async(test_short_vector_insertion,
                (h,  c,  task_id, verbose, via_Minkowski, check_module_equality, do_rfact, qary )
            ))
        for i in range(len(tasks)):
            outputs.append( tasks[i].get() )
            print(f"{i}-th task finished!")
    else:
        print("Sage 9.6+ multiprocessing not supported or nthreads==0!")
        outputs = []
        for i in range(times):
            outputs.append(test_short_vector_insertion(h,c, str(i) ,verbose, via_Minkowski, check_module_equality, do_rfact, qary ))

    for o in outputs:

        experiment_context_flag, success = o

        if success:
          succ+=1
          state_stat[experiment_context_flag][0]+=1   #gather how many successes for given experiment_context_flag
        else:
          state_stat[experiment_context_flag][1]-=1   #gather how many fails for given experiment_context_flag
    print()
    print('All experiments done in: ', perf_counter() - then)
    print((100*succ/times).n(),'% succsesfull!')

    print('v=(s0,0), s0*OK != b0 * OK,   successes: ', state_stat[0][0],' fails: ', state_stat[0][1])
    print('v=(s0,0), s0*OK  = b0 * OK ,  successes: ', state_stat[1][0],' fails: ', state_stat[1][1])

    print('v=(0,s1),                     successes: ', state_stat[2][0],' fails: ', state_stat[2][1])
    print('v=(s0,s1),                    successes: ', state_stat[3][0],' fails: ', state_stat[3][1])



h = 2**5
N = 1
res = test_short_vector_insertion_multiple_times(h, times=N, c=10**12, verbose=True,do_rfact=True)
print(res)
