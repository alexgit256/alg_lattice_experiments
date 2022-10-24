import utils

def roundoff(a):
  return utils.roundoff(a)

def butterfly(v_,s):
    return utils.butterfly(v_,s)

def ifft(v,real_value=True):
    return utils.ifft(v,real_value=True)

def minkowski_embedding(a, truncated=True):
    return utils.minkowski_embedding(a, truncated=True)

def inv_minkowski_embedding(s):
    return utils.inv_minkowski_embedding(s)

def randrange_not_null(a,b):
    return utils.randrange_not_null(a,b)

def compare_sage_versions(ver0, ver1):
    return utils.compare_sage_versions(ver0, ver1)

def scale_matrix(M):
    return utils.scale_matrix(M)

def cvp_for_fractional_lattices(B,target):
    return utils.cvp_for_fractional_lattices(B,target)

def bkz_reduce_fractional_matrix(B, block_size, verbose=False, dump=True):
    return utils.bkz_reduce_fractional_matrix(B, block_size, verbose=False, dump=True)

def short_lattice_vectors(B, nr_solutions=1,  task_id=None, verbose=verbose, approx=None, radius=0.995):
    return utils.short_lattice_vectors(B, nr_solutions, task_id, verbose, approx, radius)

def nfhnf_pari(A, Is, U=None):
    return utils.nfhnf_pari(A, Is, U)

PseudoBasis = utils.PseudoBasis
