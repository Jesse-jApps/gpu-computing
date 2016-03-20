#-*- coding: utf-8 -*-
import pyopencl as cl
import pyopencl.array 
from jinja2 import Template
import time
import numpy as np
from scipy.sparse import csr_matrix

KERNEL = Template("""
    {{header}}

    #include <pyopencl-complex.h>

    __kernel
    void complex_mat_mul(__global const {{complex_type}} *a, __global const {{complex_type}} *b, __global {{complex_type}} *res)
    {
        int row = get_local_id(0);
        int col = get_local_id(1);

        int mat_id = get_group_id(0);

        //printf("mat_id: %d, row: %d, col: %d ----- ", mat_id, row, col);

        {{complex_type}} entry = 0;
        for (int e = 0; e < {{mat_dim}}; ++e) {
            entry += a[mat_id*{{mat_size}} + row * {{mat_dim}} + e] *  b[mat_id*{{mat_size}} + e * {{mat_dim}} + col];
        }
        res[mat_id*{{mat_size}} + row * {{mat_dim}} + col] = entry;
    }
""")

def get_ctx_queue(devices=[0]):
    """
    optain context and queue for spcified devices
    """
    platform         = cl.get_platforms()[0]
    platform_devices = platform.get_devices()

    ctx = cl.Context(devices=[platform_devices[x] for x in devices])
    return (ctx, cl.CommandQueue(ctx))

data_types = {
    'cfloat_t': np.complex64,
    'cdouble_t': np.complex128,
    'float': np.float32,
    'double': np.float64
}

def render_kernel(complex_type, real_type, mat_dim):
    header = ""
    if data_types[complex_type] == np.complex128:
        header = """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #define PYOPENCL_DEFINE_CDOUBLE
        """
    templ = KERNEL.render(
        header=header,
        complex_type=complex_type,
        real_type=real_type,
        mat_dim=mat_dim,
        mat_size=mat_dim**2,
    )
    print(templ)

    return templ

complex_type = 'cdouble_t'
real_type    = 'float'
mat_dim      = 25
mats_count   = 200 # x*x

ctx, queue = get_ctx_queue()

program= cl.Program(ctx, render_kernel(complex_type, real_type, mat_dim)).build()

mats_1 = np.array(np.random.rand(mats_count**2, mat_dim, mat_dim), dtype=data_types[complex_type])
mats_2 = np.array(np.random.rand(mats_count**2, mat_dim, mat_dim), dtype=data_types[complex_type])

start = time.time()
numpy_result = np.array([np.dot(mats_1[i], mats_2[i]) for i in range(mats_count**2)])
print("numpy time: %.3f" % (time.time()-start))

a = cl.array.to_device(queue, mats_1)
b = cl.array.to_device(queue, mats_2)
c = cl.array.to_device(queue, np.zeros((mats_count**2, mat_dim, mat_dim), dtype=data_types[complex_type]))

start = time.time()
program.complex_mat_mul(queue, (mat_dim*mats_count**2, mat_dim), (mat_dim, mat_dim), a.data,b.data,c.data)
queue.finish()
print("opencl time: %.3f" % (time.time()-start))
queue.flush()
result = c.get()
print("opencl time: %.3f" % (time.time()-start))

assert np.allclose(numpy_result.flatten(), result.flatten(), atol=0), "FAIL opencl"
print("Success")