# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
    np.ulong_t index
    np.float64_t value
    np.int64_t label

cdef inline int _compare(const_void *a, const_void *b):
    cdef np.float64_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return 1
    if v >= 0: return -1

def mrr_fast_v3(np.int64_t[:] labels, np.float64_t[:] preds, np.int64_t[:] group_lengths):
    cdef int i,j,k,g,m,n
    cdef float mrrsum = 0
    cdef float n_groups = group_lengths.shape[0]
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(25*sizeof(IndexedElement))

    for k in range(25):
        order_struct[k].index = k

    i = 0

    for n in range(group_lengths.shape[0]):
        g = group_lengths[n]
        j = i + g

        for k in range(g):
            order_struct[k].value = preds[i+k]
            order_struct[k].label = labels[i+k]

        qsort(<void *> order_struct, g, sizeof(IndexedElement), _compare)

        for m in range(g):
            if order_struct[m].label == 1:
                break

        mrrsum += 1.0/(m+1.0)
        i += g

    return mrrsum / n_groups
