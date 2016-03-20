#-*- coding: utf-8 -*-
import pyopencl as cl
import pyopencl.array 
import numpy as np
from numpy import linalg
import argparse, time, os

#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")


"""
Grid limit seems to be at around 250x250=62,500 for GeForce 920M

"""

def get_ctx_queue(devices=[0]):
    """
    optain context and queue for spcified devices
    """
    platform         = cl.get_platforms()[0]
    platform_devices = platform.get_devices()

    ctx = cl.Context(devices=[platform_devices[x] for x in devices])
    return (ctx, cl.CommandQueue(ctx))


class System():
    """single junction system"""
    def __init__(self, dim, U, J):
        self.dim = dim
        self.U = U
        self.J = J

        self.h_0  = np.array(np.diagflat([(x-int(dim/2.0))**2*(U/2.0) for x in range(dim)], 0) + np.diagflat([-0.5*J]*(dim-1), 1) + np.diagflat([-0.5*J]*(dim-1), -1), dtype=np.float32)

        eigv, eigs = linalg.eig(self.h_0)

        self.eigenstates = np.array(eigs.T[np.argsort(eigv), :], dtype=np.float32)
        self.eigenvalues = np.array(np.sort(eigv), dtype=np.float32)

    def h_dr(self, w, a, t):
        return np.array(np.zeros((self.dim, self.dim)) + a*np.cos(w*t+np.pi/2.0)*(np.diagflat([1]*(self.dim-1), 1)+np.diagflat([1]*(self.dim-1), -1)), dtype=np.float32)


class Solver():
    """
    solve schrödinger equation for groundstate
    """
    CL_KERNEL = """
    #include <pyopencl-complex.h>
    #define PI 3.141592f

    __kernel 
    void calc(__global cfloat_t* states, __global const float* h_0, __global const float* w, __global const float* a, float t, float h, __global cfloat_t *sum)
    {
        int a_range_id = get_global_id(0);
        
        int w_range_id = get_global_id(1);
        int w_size     = get_global_size(1);

        int state_id = a_range_id + w_range_id*w_size;

        cfloat_t i = (cfloat_t)(0.0f,1.0f);

        cfloat_t k1[DIM];
        cfloat_t k2[DIM];
        cfloat_t k3[DIM];

        float _h;


        for (int row=0; row < DIM; row++) {
            k1[row] = 0.0f;
            for (int col=0; col < DIM; col++) {
                _h = h_0[row*DIM + col];
                if(row + 1 == col || row - 1 == col) {
                    _h += a[a_range_id]*cos(w[w_range_id]*t+(PI*0.5f));
                }

                k1[row] -= cfloat_mul(i, _h * states[state_id*DIM + col]);
            }
        }


        for (int row=0; row < DIM; row++) {
            k2[row] = 0.0f;
            for (int col=0; col < DIM; col++) {
                _h = h_0[row*DIM + col];
                if(row + 1 == col || row - 1 == col) {
                    _h += a[a_range_id]*cos(w[w_range_id]*(t+0.5f*h)+(PI*0.5f));
                }

                k2[row] -= cfloat_mul(i, _h * (0.5f*h*k1[col]+states[state_id*DIM + col]));
            }
        }

        for (int row=0; row < DIM; row++) {
            k3[row] = 0.0f;
            for (int col=0; col < DIM; col++) {
                _h = h_0[row*DIM + col];
                if(row + 1 == col || row - 1 == col) {
                    _h += a[a_range_id]*cos(w[w_range_id]*(t+h)+(PI*0.5f));
                }

                k3[row] -= cfloat_mul(i, _h * (states[state_id*DIM + col]-h*k1[col]+2.0f*h*k2[col]));
            }
        }

        sum[state_id] = 0.0f;
        for (int row=0; row < DIM; row++) {
            states[state_id*DIM+row] += h*(k1[row]/6.0f+2.0f/3.0f*k2[row]+k3[row]/6.0f);
            sum[state_id] += states[state_id*DIM+row];
        }

    }
    """

    def __init__(self, ctx, dim):
        self.ctx = ctx
        self.dim = dim
        self.runge_kutta_programm = cl.Program(ctx, self.CL_KERNEL.replace("DIM", str(dim))).build()

    @staticmethod
    def sum(state):
        """
        simple sum of elements (no physical relevance)
        """
        return np.sum(state)

    @staticmethod
    def schroedinger(state, hamiltonian):
        return -1j*np.dot(hamiltonian, state)

    def runge_kutta(self, f, state, hamiltonian, t, h):
        k_1 = f(state, hamiltonian(t))
        k_2 = f(state+np.float32(0.5)*h*k_1, hamiltonian(t+np.float32(.5)*h))
        k_3 = f(state-h*k_1+2*h*k_2, hamiltonian(t+h))

        return state + h*(np.float32(1/6.0)*k_1+np.float32(2/3.0)*k_2+np.float32(1/6.0)*k_3)
        


def calculate_numpy(system, solver, a_range, w_range, setps, h, x_value=None):
    def solve_for(w, a):
        def hamiltonian(time):
            return system.h_0 + system.h_dr(w, a, time)
        t = 0
        state = system.eigenstates[0]
        for i in range(steps):
            state = solver.runge_kutta(solver.schroedinger, state, hamiltonian, np.float32(t), np.float32(h))
            t += h
        return state

    results = np.zeros((len(a_range), len(w_range)), dtype=np.complex64)
    start_time = time.time()
    for i, a in enumerate(a_range):
        for l, w in enumerate(w_range):
            results[l][i] = solver.sum(solve_for(w, a))

    print("Numpy result after: %.3f seconds" % (time.time()-start_time))

    return (x_value, time.time()-start_time, results)

def calculate_cl(system, a_range, w_range, steps, h, x_value=None):
    start_time = time.time()

    grid_size = len(a_range)*len(w_range)
    states = np.array([system.eigenstates[0],]*grid_size, dtype=np.complex64)
    cl_states = cl.array.to_device(queue, states)
    cl_sum = cl.array.to_device(queue, np.zeros(grid_size, dtype=np.complex64))
    cl_h_0 = cl.array.to_device(queue, np.array(system.h_0, dtype=np.float32))
    cl_w_range = cl.array.to_device(queue, w_range)
    cl_a_range = cl.array.to_device(queue, a_range)

    t = 0
    for i in range(steps):
        solver.runge_kutta_programm.calc(queue, (len(a_range),len(w_range)), None, 
            cl_states.data, 
            cl_h_0.data,
            cl_w_range.data,
            cl_a_range.data,
            np.float32(t),
            np.float32(h),
            cl_sum.data
        )

        t += h

    queue.finish()
    queue.flush()

    print("OpenCL result after: %.3f seconds" % (time.time()-start_time))

    return (x_value, time.time()-start_time, cl_sum.get())


def create_file_name(platform, calc_type, x_axis, y_axis, additional=[]):
    return platform+'_'+calc_type+'_'+y_axis+'_on_'+x_axis+'_'.join(additional)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCL calculation')
    parser.add_argument('platform',  help='Platform name')
    parser.add_argument('-n', '--numpy',  help='Solve using numpy', action='store_true')
    parser.add_argument('-c', '--cl',  help='Solve using OpenCL', action='store_true')
    parser.add_argument('-co', '--compare',  help='Compare results', action='store_true')
    parser.add_argument('-ed', '--evaluatedim',  help='evaluate dim performance', action='store_true')
    parser.add_argument('-eg', '--evaluategrid',  help='evaluate grid performance', action='store_true')
    parser.add_argument('--clonly',  help='evaluate using opencl only', action='store_true')
    args = parser.parse_args()

    platform_name = args.platform

    ctx, queue = get_ctx_queue()

    #fixed
    U = 1
    J = 1
    h = 0.005

    #variable
    dim = 11
    steps = 1000

    #variable
    a_range = np.arange(1, 5, 1, dtype=np.float32)
    w_range = np.arange(1, 5, 1, dtype=np.float32)

    system = System(dim, U, J)
    solver = Solver(ctx, dim)

    if args.evaluatedim:
        print("Evaluating performance for different dimensions...")
        dim_range = np.arange(3, 41, 2)

        numpy_time_results = np.zeros((len(dim_range), 2))
        numpy_data_results = np.zeros((len(dim_range), len(a_range), len(w_range)), dtype=np.complex64)
        for i, dim in enumerate(dim_range):
            system = System(dim, U, J)
            solver = Solver(ctx, dim)

            x, y, result = calculate_numpy(system, solver, a_range, w_range, steps, h, dim)
            numpy_time_results[i] = (x,y)
            numpy_data_results[i] = result

        cl_time_results = np.zeros((len(dim_range), 2))
        cl_data_results = np.zeros((len(dim_range), len(a_range)*len(w_range)), dtype=np.complex64)
        for i, dim in enumerate(dim_range):
            system = System(dim, U, J)
            solver = Solver(ctx, dim)

            x, y, result = calculate_cl(system, a_range, w_range, steps, h, dim)
            cl_time_results[i] = (x,y)
            cl_data_results[i] = result

        print("Comparing results...")
        assert np.allclose(numpy_data_results.flatten(), cl_data_results.flatten(), atol=0.001), "Not equal"
        print("Success!")

        np.save(os.path.join(DATA, create_file_name(
            platform_name,
            'cl',
            'dim',
            'time'
        )), cl_time_results)
        np.save(os.path.join(DATA, create_file_name(
            platform_name,
            'np',
            'dim',
            'time'
        )), numpy_time_results)

        #print(np.load(os.path.join(DATA, "dim_cl_time_results.npy")))
        #print(np.load(os.path.join(DATA, "dim_numpy_time_results.npy")))

    elif args.evaluategrid:
        print("Evaluating performance for different grid sizes...")
        systems = np.arange(10, 200, 10)
        grid_ranges = np.array([np.arange(1, x, 1, dtype=np.float32) for x in systems])

        if not args.clonly:
            numpy_time_results = np.zeros((len(grid_ranges), 2))
            numpy_data_results = []
            for i, grid in enumerate(grid_ranges):
                x, y, result = calculate_numpy(system, solver, grid, grid, steps, h, len(grid)*len(grid))
                numpy_time_results[i] = (x,y)
                numpy_data_results += list(result.flatten())


        cl_time_results = np.zeros((len(grid_ranges), 2))
        cl_data_results = []
        for i, grid in enumerate(grid_ranges):
            x, y, result = calculate_cl(system, grid, grid, steps, h, len(grid)*len(grid))
            cl_time_results[i] = (x,y)
            cl_data_results += list(result.flatten())

        if args.clonly:
            np.save(os.path.join(DATA, create_file_name(
                platform_name,
                'cl',
                'systemsize',
                'time',
                ['for_%dx%d_to_%dx%d' % (systems[0], systems[0], systems[-1], systems[-1]),
                'at_dim_%d' % dim]
            )), cl_time_results)


        if not args.clonly:
            print("Comparing results...")
            assert np.allclose(numpy_data_results, cl_data_results, atol=0.001), "Not equal"
            print("Success!")

            np.save(os.path.join(DATA, create_file_name(
                platform_name,
                'cl',
                'systemsize',
                'time',
                ['for_%dx%d_to_%dx%d' % (systems[0], systems[0], systems[-1], systems[-1]),
                'at_dim_%d' % dim]
            )), cl_time_results)
            np.save(os.path.join(DATA, create_file_name(
                platform_name,
                'np',
                'systemsize',
                'time',
                ['for_%dx%d_to_%dx%d' % (systems[0], systems[0], systems[-1], systems[-1]),
                'at_dim_%d' % dim]
            )), numpy_time_results)

            #print(np.load(os.path.join(DATA, "grid_cl_time_results.npy")))
            #print(np.load(os.path.join(DATA, "grid_numpy_time_results.npy")))

    elif args.numpy:
        calculate_numpy(system, solver, a_range, w_range, steps, h, dim)

    elif args.cl:
        calculate_cl(system, a_range, w_range, steps, h, dim)


    elif args.compare:
        x, y, np_result = calculate_numpy(system, solver, a_range, w_range, steps, h, dim)
        x, y, cl_result = calculate_cl(system, a_range, w_range, steps, h, dim)

        print(np_result)

        print(cl_result)

        print("Comparing results...")
        assert np.allclose(np_result.flatten(), cl_result.flatten(), atol=0.001), "Not equal"
        print("Success!")
    else:
        parser.print_help()







