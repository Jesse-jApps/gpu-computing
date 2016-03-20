
import numpy as np
from scipy.sparse import csr_matrix
import time

def H_0_tensor(dim, J_w, J_s, U):
    h_1  = np.diagflat([(x-int(dim/2.0))**2*(U/2.0) for x in range(dim)], 0)
    h_1 += np.diagflat([-0.5*J_w]*(dim-1), 1) + np.diagflat([-0.5*J_w]*(dim-1), -1)

    h_2  = np.diagflat([(x-int(dim/2.0))**2*(U/2.0) for x in range(dim)], 0)
    h_2 += np.diagflat([-0.5*J_s]*(dim-1), 1) + np.diagflat([-0.5*J_s]*(dim-1), -1)

    correct = np.tensordot(h_1, np.identity(dim), axes=0) + np.tensordot(np.identity(dim), h_2, axes=0)

    return correct.swapaxes(1,2).reshape(dim*dim,-1)



dim = 5

m = H_0_tensor(dim,1,1,1)
r = np.array(np.random.rand(dim**2, dim**2))

#print(m)
sparse = csr_matrix(m)
#print(sparse)


#print(sparse.indptr)
#print(sparse.indices)

#srow_sizes = [sparse.indptr[i] - sparse.indptr[i-1] for i in range(1, dim**2+1, 1)]
#print(row_sizes)

start = time.time()
result = np.zeros((dim**2, dim**2))
for row in range(dim**2):
    for col in range(dim**2):

        value = 0
        for i in range(dim**2):
            value += m[row][i] * r.T[col][i]

        result[row][col] = value

print("by hand %.3f" % (time.time()-start))



start = time.time()
result_sparse = np.zeros((dim**2, dim**2))
for row in range(dim**2):
    for col in range(dim**2):

        value = 0
        for i in range(sparse.indptr[row], sparse.indptr[row+1], 1):
            col_index = sparse.indices[i]
            value += m[row][col_index] * r.T[col][col_index]

        result_sparse[row][col] = value

print("by hand sparse %.3f" % (time.time()-start))

start = time.time()
np_res = np.dot(m, r)
print("numpy %.3f" % (time.time()-start))

assert np.array_equal(np_res, result)
assert np.array_equal(np_res, result_sparse)
print("Success")



