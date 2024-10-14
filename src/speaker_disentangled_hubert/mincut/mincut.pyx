cimport cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt
from cython.operator import postincrement as inc
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from "float.h":
    double DBL_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def min_cut(np.ndarray ssm, Py_ssize_t K, Py_ssize_t max_frames):
    cdef Py_ssize_t N = ssm.shape[0] + 1

    cdef double[:,::1] C = np.ones((N, K), dtype=np.float32, order="C") * DBL_MAX
    cdef int[:,::1] B = np.zeros((N, K), dtype=np.int32)

    C[0,0] = 0.

    cdef list temp, obj
    cdef Py_ssize_t i, j, k, ind
    for i in range(1,N):
        start = max(0, i - max_frames)
        temp = [(ssm[j:i, j:i].sum() / 2., ssm[j:i, :j].sum() + ssm[j:i, i:].sum()) for j in range(start, i)]
        for k in range(1,min(K,i+1)):
            obj = [C[j, k-1] + item[1]/(item[0]+item[1]) for j, item in zip(range(start, i), temp)]
            ind = np.argmin(obj)
            B[i,k] = ind + start
            C[i,k] = obj[ind]
    
    # backtrack
    cdef list boundary = []
    cdef Py_ssize_t prev_b = N - 1
    cdef list loop = list(range(1,K))[::-1]
    boundary.append(prev_b)
    for k in loop:
        prev_b = B[prev_b,k]
        boundary.append(prev_b)
    boundary = boundary[::-1] # reverse
    return boundary
