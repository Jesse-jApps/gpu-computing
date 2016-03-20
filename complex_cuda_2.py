#-*- coding: utf-8 -*-
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from jinja2 import Template
import time
import numpy as np
from scipy.sparse import csr_matrix

"""
IDEA:
* test the use of complex data types
* compare double and single precision
* test performance for matrix multiplication in a single thread 

Input are 2 buffers that each contain multiple matrices
"""

KERNEL = Template("""
    #include <stdio.h>
    #include <pycuda-complex.hpp>

    typedef pycuda::complex<float> scmplx;
    typedef pycuda::complex<double> dcmplx;

    __global__ void complex_mat_mul(const {{complex_type}} *a, const {{complex_type}} *b, {{complex_type}} *res)
    {
        int row = blockIdx.x;
        int col = blockIdx.y;

        int mat_id = threadIdx.x * blockDim.x + threadIdx.y; 

        //printf("mat_id: %d, row: %d, col: %d ----- ", mat_id, row, col);
        //printf("block x: %d, block y: %d griddim: %d ----- ", blockIdx.x, blockIdx.y, gridDim.x);



        {{complex_type}} entry = 0;
        for (int e = 0; e < {{mat_dim}}; ++e) {
            entry += a[mat_id*{{mat_dim}}*{{mat_dim}} + row * {{mat_dim}} + e] *  b[mat_id*{{mat_dim}}*{{mat_dim}} + e * {{mat_dim}} + col];
        }
        res[mat_id*{{mat_dim}}*{{mat_dim}} + row * {{mat_dim}} + col] = entry;


        
    }


""")

data_types = {
    'scmplx': np.complex64,
    'dcmplx': np.complex128,
    'float': np.float32,
    'double': np.float64
}

def render_kernel(complex_type, real_type, mat_dim, block, gird):
    templ = KERNEL.render(
        complex_type=complex_type,
        real_type=real_type,
        mat_dim=mat_dim,
        blockDim_x=block[0],
        blockDim_y=block[1]
    )
    print(templ)

    return templ

complex_type = 'dcmplx'
real_type    = 'double'
mat_dim      = 25
mats_count   = 16

block = (int(np.sqrt(mats_count)),int(np.sqrt(mats_count)),1)
grid  = (mat_dim,mat_dim)

program = SourceModule(render_kernel(complex_type, real_type, mat_dim, block, grid))


complex_mat_mul = program.get_function("complex_mat_mul")


mats_1 = np.array(np.random.rand(mats_count, mat_dim, mat_dim), dtype=data_types[complex_type])
mats_2 = np.array(np.random.rand(mats_count, mat_dim, mat_dim), dtype=data_types[complex_type])
result = np.zeros((mats_count, mat_dim, mat_dim), dtype=data_types[complex_type])

start = time.time()
numpy_result = np.array([np.dot(mats_1[i], mats_2[i]) for i in range(mats_count)])
print("numpy time: %.3f" % (time.time()-start))

start = time.time()
scipy_result = np.array([csr_matrix(mats_1[i]).dot(mats_2[i]) for i in range(mats_count)])
print("scipy time: %.3f" % (time.time()-start))

a = drv.In(mats_1)
b = drv.In(mats_2)
c = drv.Out(result)


start = time.time()
complex_mat_mul(a, b, c,
    block=block, 
    grid=grid
)
print("cuda time: %.3f" % (time.time()-start))

assert np.array_equal(numpy_result, result), "FAIL"
assert np.array_equal(numpy_result, scipy_result), "FAIL"
print("Success")