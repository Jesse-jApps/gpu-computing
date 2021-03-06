#-*- coding: utf-8 -*-
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from jinja2 import Template
import time
import numpy as np
from scipy.sparse import csr_matrix

KERNEL = Template("""
    #include <stdio.h>
    #include <pycuda-complex.hpp>

    typedef pycuda::complex<float> scmplx;
    typedef pycuda::complex<double> dcmplx;

    __global__ void complex_mat_mul(const {{complex_type}} *a, const {{complex_type}} *b, {{complex_type}} *res)
    {
        int row = threadIdx.y;
        int col = threadIdx.x;

        int mat_id = blockIdx.x * gridDim.x + blockIdx.y;

        //printf("mat_id: %d, row: %d, col: %d ----- ", mat_id, row, col);


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
mats_count   = 100 # x*x

block = (mat_dim,mat_dim,1)
grid  = (mats_count,mats_count)

program = SourceModule(render_kernel(complex_type, real_type, mat_dim, block, grid))

complex_mat_mul = program.get_function("complex_mat_mul")

mats_1 = np.array(np.random.rand(mats_count**2, mat_dim, mat_dim), dtype=data_types[complex_type])
mats_2 = np.array(np.random.rand(mats_count**2, mat_dim, mat_dim), dtype=data_types[complex_type])
result = np.zeros((mats_count**2, mat_dim, mat_dim), dtype=data_types[complex_type])

start = time.time()
numpy_result = np.array([np.dot(mats_1[i], mats_2[i]) for i in range(mats_count**2)])
print("numpy time: %.3f" % (time.time()-start))



a = drv.In(mats_1)
b = drv.In(mats_2)
c = drv.Out(result)

start = time.time()

complex_mat_mul(a, b, c,
    block=block, 
    grid=grid
)
print("cuda time: %.3f" % (time.time()-start))


assert np.array_equal(numpy_result.flatten(), result.flatten()), "FAIL"
print("Success")