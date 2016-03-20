#-*- coding: utf-8 -*-
import pyopencl as cl
import pyopencl.array 
import numpy as np
from numpy import linalg
import argparse, time, os
from scipy.sparse import csr_matrix

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from numpy.linalg import det

#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

DTYPE = np.float32


def get_ctx_queue(devices=[0]):
    """
    optain context and queue for spcified devices
    """
    platform         = cl.get_platforms()[0]
    platform_devices = platform.get_devices()

    ctx = cl.Context(devices=[platform_devices[x] for x in devices])
    return (ctx, cl.CommandQueue(ctx))



def create_sparse_matrix(dim):
    return np.array(np.zeros((dim, dim)) + np.diag([1]*dim) + np.diag([1]*(dim-1), -1) + np.diag([1]*(dim-1), 1), dtype=DTYPE)

    return np.array(np.zeros((dim, dim)) + np.diag(np.random.rand(dim)) + np.diag(np.random.rand(dim-1), -1) + np.diag(np.random.rand(dim-1), 1), dtype=DTYPE)

def create_random_matrix(dim):
	return np.array(np.random.rand(dim, dim), dtype=DTYPE)


def np_evaluate_sparse(sparse_mat, mat):
	A = csr_matrix(sparse_mat)

	return A.dot(mat)

def np_evaluate(A, B):
	return A.dot(B)



class CudaCalculator(object):
    KERNEL = """
        #include <stdio.h>
        //#include <cuComplex.h>


        __global__ void simple_multiply(const float *a, const float *b, float *res)
        {
            float Cvalue = 0;

            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            //printf("%d, ", threadIdx.x);

            if(row > $dim$ || col > $dim$) return;

            for (int e = 0; e < $dim$; ++e) {
                Cvalue += a[row * $dim$ + e] *  b[e * $dim$ + col];
            }
           
            res[row * $dim$ + col] = Cvalue;
            
        }


        """
    def __init__(self, dim):
        self.dim = dim

        self.mod = SourceModule(CudaCalculator.KERNEL.replace("$dim$", str(self.dim)))
        self.cu_simple_multiply = self.mod.get_function("simple_multiply")



        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse matrix calculation')
    parser.add_argument('platform',  help='Platform name')
    parser.add_argument('-np', '--np', action='store_true')
    parser.add_argument('-nps', '--npsparse', action='store_true')
    parser.add_argument('-c', '--compare', action='store_true')
    parser.add_argument('-cu', '--cuda', action='store_true')


    args = parser.parse_args()

    number_of_evaluations = 0

    if args.np: 
        number_of_evaluations += 1
        np_index = number_of_evaluations - 1

    if args.npsparse: 
        number_of_evaluations += 1
        npsparse_index = number_of_evaluations - 1

    if args.cuda: 
        number_of_evaluations += 1
        cuda_index = number_of_evaluations - 1



    platform_name = args.platform

    ctx, queue = get_ctx_queue()

    size_range = np.arange(1000, 2000, 1000)

    results = np.zeros((number_of_evaluations, len(size_range)), dtype=DTYPE)

    sparse_mat = []
    mat = []
    for dim in size_range:
    	sp = create_sparse_matrix(dim)
    	sparse_mat.append(sp)

    	mt = create_random_matrix(dim)
    	mat.append(mt)


    if args.np:
        for i, _l in enumerate(size_range):
            start_time = time.time()
            result = np_evaluate(sparse_mat[i], mat[i])
            print("np %.3f" % (time.time() - start_time))
            results[np_index][i]  = np.trace(result)


    if args.npsparse:
        for i, _l in enumerate(size_range):
            start_time = time.time()
            result = np_evaluate_sparse(sparse_mat[i], mat[i])
            print("scipy %.3f" % (time.time() - start_time))
            results[npsparse_index][i] = np.trace(result)


    if args.cuda:

        def find_divider(number):
            div = None
            i = 32.0
            while not div:
                if (float(number) / i).is_integer():
                    div = i
                i -= 1.0

            return div
        

        for i, _l in enumerate(size_range):
            calc = CudaCalculator(_l)

            result = np.zeros_like(mat[i])
            a = drv.In(sparse_mat[i])
            b = drv.In(mat[i])
            c = drv.Out(result)

            div = int(find_divider(_l))

            print("start....")
            start_time = time.time()
            calc.cu_simple_multiply(
                a,
                b,
                c,
                block=(div,div,1), 
                grid=(_l/div, _l/div)
            )
            print("cuda %.3f" % (time.time() - start_time))
            results[cuda_index][i] = np.trace(result)


    if args.compare:
        for res in results:
            assert np.allclose(res.flatten(), results[0].flatten(), atol=0.001)
            print("Results are equal")


