import sys, time
from utils_wrapper import PseudoBasis, compare_sage_versions, scale_matrix, roundoff
from keflll_utils import FieldInfo, descend_rank2_matrix, enorm_vector_over_numfield, ascend, enorm_numfield
from keflll_utils import rfact_herm, invertibles
from arakelov import arakelov_rand_walk, rand_p_ideal, bound_on_B, steps_num, nearest_P_smooth_number

from multiprocessing import Pool, cpu_count
from sage import version
import random as rnd
from time import perf_counter

from fpylll import IntegerMatrix, GSO
from fpylll.tools.quality import basis_quality

from sage.misc.randstate import current_randstate

import multiprocessing
import sage.parallel.multiprocessing_sage
from multiprocessing import Pool, Manager, cpu_count


def rand_unit(K, c=4):
    z = K.gen()
    OK = K.ring_of_integers()
    U = [OK( (1-z^(2*i+1))/(1-z) ) for i in range(K.degree()/2) ]
    u = 1
    for t in U:
        u*=t**randrange(-c,c+1)
    return u

def gen_unimod_matrix(K,n, rng=[-14,15], density=0.99):
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

def test_short_vector_insertion(h,init_ideal_norm=32,task_id=None, verbose=False, via_Minkowski=False, check_module_equality=False, do_rfact=False, qary=None, randomize_arakelov = True):
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
        print("Warning! euler_phi(h)>36: program might work extremely slow!")

    K.<z> = CyclotomicField(h)
    d = K.degree()
    OK = K.ring_of_integers()

    text_output = ''

    """
    B is the matrix of the random module. It is aslo the pseudomatrix of pseudobasis
    of the same module with ideals that are OK.
    """
    set_random_seed(hash(task_id)+int(round(2**64*perf_counter())))   #rerandomize, because this is ran in parallel
    randrange = current_randstate().python_random().randrange

    while True:  #resample modules untill the slope is negative
        B = matrix(K,[
            [OK.random_element(-6,7),     OK.random_element(-6,7)],    #generate the small matrix
            [OK.random_element(-8,9),     OK.random_element(-8,9)]
        ])

        detB = det(B)
        if detB==0:   #if rank<2 then resample
            continue
        U, W = gen_unimod_matrix( K, 2 ), gen_unimod_matrix( K, 2 )
        assert U!= W, "U and W are the same. Check random!"
        B = U*B*W
        I = rand_p_ideal( K, init_ideal_norm )
        #print('checking if I is_principal...')
        #print( f"Debug: {I.is_principal(proof=False)}" )

        assert log( h, 2 ) in ZZ, "Arakelov walks not supported for non-power-of-2 CyclotomicField"
        if verbose:
            print("Perfoming Arakelov walks. This might take a while...")
            sys.stdout.flush()

        """
        Below is the Bach bound on the norms of ideals. Note: it is < 3*10^11 in practice
        (3*10^11 is according to https://eprint.iacr.org/2020/297.pdf requirenments)
        """
        disc = 2^(d*log(d,2))   #the discriminant of pow-of-2 cyclotomic pield
        bach_bound = ceil( 12*ln(abs(disc))^2 )   #according to https://arxiv.org/pdf/1607.02430.pdf
        #bound = min( bach_bound,bound_on_B(d) ) #suggested_bound
        #N = steps_num(d) if d<32 else steps_num(d) - ceil( log(d,2)+1 ) # N_bound is computed differently after the 32nd cyclotomic field
        #s = 1/log(d,2)^2
        bound = 150000
        N=3
        s=0.08

        if verbose:
            print(f"Starting Arakelov walk with B, s, N = {bound, s, N}")
            sys.stdout.flush()
        then = time.perf_counter()
        I = arakelov_rand_walk( I, bound, s=s, N=N, normalize=True , smooth_unit=True )  #Randomize the ideal of pseudobasis
        dt = time.perf_counter()-then
        J = 1/I
        if verbose:
            print(f"Arakelov done in {dt}")
        sys.stdout.flush()

        Is = [J, I]  #Force the module (lattice) to be free, thus we'll have a basis

        pseudomatrix = PseudoBasis(B, Is, hnf_reduced=False)  #construct the pseudobasis object
        det_stash = pseudomatrix.det()
        if verbose:
          print(f"Making the module free. This might take a while... id={task_id}, bit hardness={pseudomatrix.bitlength()}")
          sys.stdout.flush()
        then = time.perf_counter()
        pseudomatrix.make_almost_free()   #obtain the OK basis of the lattice
        assert abs( ln(abs(det_stash)) - ln(abs(pseudomatrix.det())) ) < 0.5, f"Making module free changed determinant! {abs( ln(abs(det_stash)) - ln(abs(pseudomatrix.det())) )}"
        print(f"Free module obtained in {dt}.")
        sys.stdout.flush()
        assert all( [elem == Ideal(K,K(1)) for elem in Is ] ), "Failed to construct an OK basis"

        """
        Below we take the R-factor and round it to 384 bits to make it feasible for BKZ
        - - - - - - - - - - - -
        print("Computing R-factor...")
        R = rfact_herm( pseudomatrix.A )  #construct the R-factor of lattice basis
        print("R-factor computed")
        R *= 2**384  #R rounds with that precision. Must be << 2**511.
        R = matrix( K, [
          [roundoff(R[i,j]) for j in range(R.ncols())] for i in range(R.nrows())
        ]) *2**-384
        if det(R)==0:
            continue
        pseudomatrix = PseudoBasis( R, [ideal(K(1)) for i in range(R.nrows())], hnf_reduced=False )  #Obtain the lattice with R factor as defining matrix
        """

        print(f" - - - ________________________________ BITLENGTH: {pseudomatrix.bitlength()} ________________________________- - -")
        if pseudomatrix.bitlength() > 511:  #once bitlength of coefficients exceeds 511 bits, fpylll crashes.
            print( "WARNING! BKZ ATTEMPTED TO REDUCE >511 BIT MATRIX!" )
            continue    #if the bitlenght of the Z-basis coefficients is not feasible by fpylll bkz, resample
        #print( pseudomatrix.A.numerical_approx())

        dt = time.perf_counter() - then

        if verbose:
            print(f"The module is now free... id={task_id}")

        M = pseudomatrix.express_as_Q_module()
        M, l = scale_matrix( M )
        M = matrix(ZZ,M)
        print("LLL...")
        M = M.LLL()
        M = IntegerMatrix.from_matrix(M)
        g = GSO.Mat(M)
        g.update_gso()
        lll_slope = basis_quality(g)["/"]
        if lll_slope>-0.01:
            print(f"WARNING! SLOPE = {lll_slope}>0! RESAMPLING...")
            continue

        break #if we made it here, we're good to go

    #note! this one returns coords!
    if not via_Minkowski:
        v = pseudomatrix.short_elements(nr_solutions=1, algebraic_form=True,task_id=task_id, verbose=verbose)[0]
    else:
        v = pseudomatrix.short_elements_via_Minkowski(nr_solutions=1, algebraic_form=True,task_id=task_id, verbose=verbose)[0]

    #This one is sanity check.
    assert v in pseudomatrix, f"Found vector not in module!"

    ab = pseudomatrix.A.solve_left(v)

    s0 = v[0]
    s1 = v[1]


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
    We have 4 endstates of the experiment:
    0: v is in the form (s0,0) and (s0)!=(b0)
    1: v is in the form (s0,0) and (s1)==(b0)
    2: v is in the form (s0,s1)     #just for curiosity
    3: v is in the form (0,s1)      #just for curiosity

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
    print(f"experiment_context_flag = {experiment_context_flag} at id= {task_id}")

    try:
        mu, nu = add_to_one(ab[0],ab[1])
        nu = -nu  #we are searching for the solutions of a*mu-b*nu = 1
        OK(mu), OK(nu)  #if not integral, we\'ll get error

        assert OK(ab[0]*mu-ab[1]*nu).is_unit(), f"Not a Bezout solution! " + str(ab[0]*mu+ab[1]*nu) + str((mu,nu))
        U = matrix(K,[
        [ab[0],ab[1]],
        [nu, mu]
        ])
        if verbose:
            print(f"Lift sucsess! Task id: {task_id}")
    except Exception as err:
        if verbose:
            print(f"Geuclide failed at task_id= {task_id}" )
            #print(f"_______________{err}_____________gcd={gcd(ab[0].norm(),ab[1].norm())}")
        return experiment_context_flag, False
    sys.stdout.flush()

    if check_module_equality:  #if we want to check if the module we obtained is the same one
        M = U*pseudomatrix.A
        pseudomatrix_post = PseudoBasis(M, [Ideal(K(1)) for i in range(M.nrows())], hnf_reduced=False)
        try:
            assert pseudomatrix_post == pseudomatrix , f"Modules are not equal!"
        except AssertionError as err:
            if verbose:
                print(err)
                print(U)
                print(pseudomatrix_post.det().n(),pseudomatrix.det().n() )
            sys.stdout.flush()
            return experiment_context_flag, False
    sys.stdout.flush()
    return experiment_context_flag, True

def test_short_vector_insertion_multiple_times(
    h,                #conductor of the field
    times=5,          #number of the experiments
    init_ideal_norm=32,          #initial ideal I's norm is bounded by this value
    verbose = False,
    via_Minkowski = False,  #use Minkowski embedding
    check_module_equality = False,  #check if the module after the insertion is equal to the original one. Relatively slow and always should pass if no bugs.
    do_rfact = False,        #do rfact_herm on the matrix to simulate KEF-LLL routin
    nthreads = 1,           #amount of threads to utilize (<2 if not using multithreading at all)
    qary = None ,            #if not None, should be an integer q. Then the module is (q,0)*OK + ([h mod q],1)*OK
    randomize_arakelov = True
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
            tasks.append(  pool.apply_async(test_short_vector_insertion,
                (h,  init_ideal_norm,  i, verbose, via_Minkowski, check_module_equality, do_rfact, qary, randomize_arakelov )
            ))
        for i in range(len(tasks)):
            outputs.append( tasks[i].get() )
            print(f"{i}-th task finished!")
            sys.stdout.flush()
    else:
        print("Sage 9.6+ multiprocessing not supported or nthreads==0!")
        outputs = []
        for i in range(times):
            outputs.append(test_short_vector_insertion(h,init_ideal_norm, str(i) ,verbose, via_Minkowski, check_module_equality, do_rfact, qary, randomize_arakelov ))

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


"""
The functionn below works as test_short_vector_insertion_multiple_times, but if stdout_path is not None, it dumps the output and stderr to the specified files
"""
def launch_test_short_vector_insertion_multiple_times(
      h,                #conductor of the field
      times=5,          #number of the experiments
      init_ideal_norm=32,    #initial ideal I's norm is bounded by this value. We coustruct PseudoBasis [ B, [I, 1/J] ] so the norm(I) <= init_ideal_norm
      verbose = False,
      via_Minkowski = False,  #use Minkowski embedding
      check_module_equality = False,  #check if the module after the insertion is equal to the original one. Relatively slow and always should pass if no bugs.
      do_rfact = False,        #do rfact_herm on the matrix to simulate KEF-LLL routin
      nthreads = 1,           #amount of threads to utilize (<2 if not using multithreading at all)
      qary = None,             #if not None, should be an integer q. Then the module is (q,0)*OK + ([h mod q],1)*OK
      stdout_path = 0         #if None, no logging; if 0, path is default one; str considered as file name
    ):
    if stdout_path is None:
        test_short_vector_insertion_multiple_times( h, times, init_ideal_norm, verbose, via_Minkowski, check_module_equality, do_rfact, nthreads, qary )
    else:
        file_num = randrange(2**31)
        if stdout_path == 0:
            stdout_path = f"tmr_output_{h}_{file_num}.txt"
        #print(f"Dumping experiments to the file {stdout_path} ...")
        stdout_ = open(stdout_path, 'w')
        with stdout_ as sys.stdout:
            try:
                test_short_vector_insertion_multiple_times( h, times, init_ideal_norm, verbose, via_Minkowski, check_module_equality, do_rfact, nthreads, qary )
            except Exception as err:
                print("Program crashed!")
                print(err)
            sys.stderr.flush()
            sys.stdout.flush()
            time.sleep(float(0.1))
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__    #close the stream


        #print("All completed sucsessfully!")


h = 64
N = 100
#launch_test_short_vector_insertion_multiple_times(h, times=N, init_ideal_norm=32, verbose=True, nthreads=50)
