# alg_lattice_experiments
In this repository we present a code that conducts experinments regarding how often we cannot insert the shortest vector of the algebraic module into the basis.

Script `test_principal_ideals_gens.sage` contains function `test_field(field_conductor,p_max=10, p=3, verbose=False)` that checks `field_conductor`-th cyclotomic field in the following way. Starting with the next prime after `p`, it generates ideals above consequent primes up to `p_max` and checks if they are generated by their nonzero element.

File `test_rank2_modules.sage` contains function `test_short_vector_insertion_multiple_times`. Among others, next parameters are the most helpful ones:

    h - the conductor of the field as in test_principal_ideals_gens;
    times=5 - the number of the experiments
    via_Minkowski - use Minkowski embedding for the CVP
    nthreads = 1 - the amount of threads to utilize (<2 if not using multithreading at all)
    qary - if not None, should be an integer q. Then the module is (q,0)*OK + ([h mod q],1)*OK
    
It constructs `times` rank-2 modules over the number field, embeds them (using the coefficient embedding if `via_Minkowski` is False) and run CVP to obtain the shortest nonzeero vector of the module. It then tries to insert it into the algeebraic basis as in [KEF16] and gathers the statistics of fails / sucsesses.

File `test_Qembedding.sage` is intended to check if the coefficient embedding is performed correctly. 
