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
    void complex_mat_mul(__global const {{complex_type}} *a, __global const {{complex_type}} *b, __global {{complex_type}} *res,__global const int *row_data, __global const int *col_data)
    {
        int row = get_local_id(1);
        int col = get_local_id(0);

        int mat_id = get_group_id(0) * get_num_groups(0) + get_group_id(1);

        //printf("mat_id: %d, row: %d, col: %d ----- ", mat_id, row, col);

        {{complex_type}} entry = 0;
        for (int e = row_data[row]; e < row_data[row+1]; ++e) {
            entry += a[mat_id*{{mat_size}} + row * {{mat_dim}} + col_data[e]] *  b[mat_id*{{mat_size}} + col_data[e] * {{mat_dim}} + col];
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

def H_0_tensor(dim, J_w, J_s, U):
    h_1  = np.diagflat([(x-int(dim/2.0))**2*(U/2.0) for x in range(dim)], 0)
    h_1 += np.diagflat([-0.5*J_w]*(dim-1), 1) + np.diagflat([-0.5*J_w]*(dim-1), -1)

    h_2  = np.diagflat([(x-int(dim/2.0))**2*(U/2.0) for x in range(dim)], 0)
    h_2 += np.diagflat([-0.5*J_s]*(dim-1), 1) + np.diagflat([-0.5*J_s]*(dim-1), -1)

    correct = np.tensordot(h_1, np.identity(dim), axes=0) + np.tensordot(np.identity(dim), h_2, axes=0)

    return correct.swapaxes(1,2).reshape(dim*dim,-1)

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
        mat_size=mat_dim**2
    )
    print(templ)

    return templ

complex_type = 'cdouble_t'
real_type    = 'double'
mat_dim      = 25
mats_count   = 220 # x*x

ctx, queue = get_ctx_queue()

program= cl.Program(ctx, render_kernel(complex_type, real_type, mat_dim)).build()



mats_1 = np.array([H_0_tensor(int(np.sqrt(mat_dim)), 1,1,float(u)/(mats_count**2)*10) for u in range(mats_count**2)], dtype=data_types[complex_type])
mats_2 = np.array(np.random.rand(mats_count**2, mat_dim, mat_dim), dtype=data_types[complex_type])

start = time.time()
numpy_result = np.array([np.dot(mats_1[i], mats_2[i]) for i in range(mats_count**2)])
print("numpy time: %.3f" % (time.time()-start))

#start = time.time()
#scipy_result = np.array([csr_matrix(mats_1[i]).dot(mats_2[i]) for i in range(mats_count**2)])
#print("scipy time: %.3f" % (time.time()-start))



a = cl.array.to_device(queue, mats_1)
b = cl.array.to_device(queue, mats_2)
c = cl.array.to_device(queue, np.zeros((mats_count**2, mat_dim, mat_dim), dtype=data_types[complex_type]))

sparse = csr_matrix(mats_1[1])
#print(sparse.indptr)
#print(sparse.indices)
row_data = cl.array.to_device(queue, sparse.indptr)
col_data = cl.array.to_device(queue, sparse.indices)

start = time.time()
program.complex_mat_mul(queue, (mats_count*mat_dim, mats_count*mat_dim), (mat_dim, mat_dim), a.data,b.data,c.data, row_data.data, col_data.data)
queue.finish()
print("opencl time: %.3f" % (time.time()-start))
queue.flush()
result = c.get()
print("opencl time: %.3f" % (time.time()-start))

assert np.allclose(numpy_result.flatten(), result.flatten(), atol=0), "FAIL opencl"
#assert np.allclose(numpy_result.flatten(), scipy_result.flatten(), atol=0), "FAIL scipy"
print("Success")