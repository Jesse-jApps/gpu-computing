#-*- coding: utf-8 -*-
import pyopencl as cl
import pyopencl.array 
from jinja2 import Template
import time, os
import numpy as np
from scipy.sparse import csr_matrix
from numpy import linalg as LA

BASE = os.path.dirname(os.path.abspath(__file__))

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def load_file(file_path):
    with open(BASE + '/' + file_path, 'r') as content_file:
        return content_file.read()


RUNGE_KUTTA_KERNEL = Template(load_file('rk3_cl.c'))

def get_ctx_queue(devices=[0]):
    """
    optain context and queue for spcified devices
    """
    platform         = cl.get_platforms()[0]
    platform_devices = platform.get_devices()

    ctx = cl.Context(devices=[platform_devices[x] for x in devices])
    return (ctx, cl.CommandQueue(ctx))

def density_matrix(T, eigv, eigs, a_state_index=0):
    dim = len(eigv)
    b   = 1/float(T)
    z   = np.sum([np.exp(-b*e) for e in eigv])

    p_initial     = 1/float(z) * np.sum([np.exp(-b*eigv[i])*np.outer(eigs[i],eigs[i]) for i in range(dim)], axis=0)
    p_groundstate = np.outer(eigs[a_state_index],eigs[a_state_index])

    return (p_initial, p_groundstate)

def H_0_tensor(dim, J_w, J_s, U):
    h_1  = np.diagflat([(x-int(dim/2.0))**2*(U/2.0) for x in range(dim)], 0)
    h_1 += np.diagflat([-0.5*J_w]*(dim-1), 1) + np.diagflat([-0.5*J_w]*(dim-1), -1)

    h_2  = np.diagflat([(x-int(dim/2.0))**2*(U/2.0) for x in range(dim)], 0)
    h_2 += np.diagflat([-0.5*J_s]*(dim-1), 1) + np.diagflat([-0.5*J_s]*(dim-1), -1)

    correct = np.tensordot(h_1, np.identity(dim), axes=0) + np.tensordot(np.identity(dim), h_2, axes=0)

    return correct.swapaxes(1,2).reshape(dim*dim,-1)

def render_kernel(kernel, complex_type, real_type, mat_dim, kwargs):
    header = ""
    if data_types[complex_type] == np.complex128:
        header = """
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
            #define PYOPENCL_DEFINE_CDOUBLE
        """
    templ = kernel.render(
        header=header,
        complex_type=complex_type,
        real_type=real_type,
        mat_dim=mat_dim,
        mat_size=mat_dim**2,
        **kwargs
    )
    print(templ)

    return templ

data_types = {
    'cfloat': np.complex64,
    'cdouble': np.complex128,
    'float': np.float32,
    'double': np.float64
}

complex_type = 'cdouble'
real_type    = 'double'
mat_dim      = 4
mats_count   = 2 # x*x
h            = 0.005
U            = 1
J_w          = 1
J_s          = 1
T            = 1

aw_range     = (1*J_w, 30*J_w)
as_range     = (1*J_s, 30*J_s)

ww_range     = (1*np.sqrt(J_w*U), 30*np.sqrt(J_w*U))
ws_range     = (1*np.sqrt(J_s*U), 30*np.sqrt(J_s*U))


ctx, queue = get_ctx_queue()
program = cl.Program(ctx, render_kernel(
    RUNGE_KUTTA_KERNEL, complex_type, real_type, mat_dim, {
        'step_size': h,
        'b1': '0.16666667',
        'b2': '0.66666666'
    }
)).build()

H = H_0_tensor(int(np.sqrt(mat_dim)), J_w, J_s, U)

eigv, eigs = LA.eig(H)
eigs = eigs.T

eigs = eigs[np.argsort(eigv), :]
eigv = np.sort(eigv)

density, groundsate = density_matrix(T, eigv, eigs)


hamiltons = np.array([H.copy() for x in range(mats_count**2)], dtype=data_types[complex_type])
densities = np.array([density.copy() for x in range(mats_count**2)], dtype=data_types[complex_type])


h = cl.array.to_device(queue, hamiltons)
p = cl.array.to_device(queue, densities)

Aw = cl.array.to_device(queue, np.arange(aw_range[0], aw_range[1], (aw_range[1] - aw_range[0]) / mats_count))
As = cl.array.to_device(queue, np.arange(as_range[0], as_range[1], (as_range[1] - as_range[0]) / mats_count))

ww = cl.array.to_device(queue, np.arange(ww_range[0], ww_range[1], (ww_range[1] - ww_range[0]) / mats_count))
ws = cl.array.to_device(queue, np.arange(ws_range[0], ws_range[1], (ws_range[1] - ws_range[0]) / mats_count))


sparse = csr_matrix(H)
row_data = cl.array.to_device(queue, sparse.indptr)
col_data = cl.array.to_device(queue, sparse.indices)

start = time.time()

steps = 1
t = 0

#for i in range(steps):
program.runge_kutta(queue, (mats_count*mat_dim, mats_count*mat_dim), (mat_dim, mat_dim), 
    h.data, p.data,
  #  row_data.data, col_data.data, 
  #  Aw.data, As.data, ww.data, ws.data, 
    np.int32(t)
)
#queue.finish()
print("opencl time: %.3f" % (time.time()-start))

t += h

    
queue.flush()
result = p.get()
print("opencl time: %.3f" % (time.time()-start))
