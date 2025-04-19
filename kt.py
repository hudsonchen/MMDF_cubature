import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
import jax
import jax.numpy as jnp
import numpy as np
import sys
import pwd
from functools import partial
import argparse
from mmd_flow.distributions import Distribution, Empirical_Distribution
from mmd_flow.kernels import gaussian_kernel
import mmd_flow.utils
from goodpoints import kt , compress
import time
import pickle
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

if pwd.getpwuid(os.getuid())[0] == 'zongchen':
    os.chdir('/home/zongchen/mmd_flow_cubature/')
    sys.path.append('/home/zongchen/mmd_flow_cubature/')
elif pwd.getpwuid(os.getuid())[0] == 'ucabzc9':
    os.chdir('/home/ucabzc9/Scratch/mmd_flow_cubature/')
    sys.path.append('/home/ucabzc9/Scratch/mmd_flow_cubature/')
else:
    pass

def get_config():
    parser = argparse.ArgumentParser(description='mmd_flow_cubature')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--kernel', type=str, default='Gaussian')
    parser.add_argument('--step_size', type=float, default=0.1) # Step size will be rescaled by lmbda, the actual step size = step size * lmbda
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--bandwidth', type=float, default=0.1)
    parser.add_argument('--step_num', type=int, default=10000)
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--inject_noise_scale', type=float, default=0.0)
    parser.add_argument('--integrand', type=str, default='neg_exp')
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"kernel_thinning/{args.dataset}_dataset/{args.kernel}_kernel/"
    args.save_path += f"__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__particle_num_{2 ** int(args.m)}__inject_noise_scale_{args.inject_noise_scale}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


def kernel_eval(x, y, params_k):
    """Returns matrix of kernel evaluations kernel(xi, yi) for each row index i.
    x and y should have the same number of columns, and x should either have the
    same shape as y or consist of a single row, in which case, x is broadcasted 
    to have the same shape as y.
    """
    if params_k["name"] in ["gauss", "gauss_rt"]:
        k_vals = np.sum((x-y)**2,axis=1)
        scale = -.5/params_k["var"]
        return(np.exp(scale*k_vals))
    
    raise ValueError("Unrecognized kernel name {}".format(params_k["name"]))


def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    kernel = gaussian_kernel(args.bandwidth)
    if args.dataset == 'gaussian':
        distribution = Distribution(kernel=kernel, means=jnp.array([[0.0, 0.0]]), covariances=jnp.eye(2)[None, :], 
                                    integrand_name=args.integrand, weights=None)
        d = 2
    elif args.dataset == 'mog':
        covariances = jnp.load('data/mog_covs.npy')
        means = jnp.load('data/mog_means.npy')
        k = 20
        d = 2
        weights = jnp.ones(k) / k
        distribution = Distribution(kernel=kernel, means=means, covariances=covariances, integrand_name=args.integrand, weights=weights)
    elif args.dataset == 'house_8L':
        data = np.genfromtxt('data/house_8L.csv', delimiter=',', skip_header=1)[:,:-1]
        d = data.shape[1]
        distribution = Empirical_Distribution(kernel=kernel, samples=data, integrand_name=args.integrand)
    elif args.dataset == 'elevators':
        data = np.genfromtxt('data/elevators.csv', delimiter=',', skip_header=1)[:,:-1]
        d = data.shape[1]
        distribution = Empirical_Distribution(kernel=kernel, samples=data, integrand_name=args.integrand)
    else:
        raise ValueError('Dataset not recognized!')

    d = int(2)
    var = jnp.square(float(args.bandwidth))
    params_k_swap = {"name": "gauss", "var": var, "d": int(d)}
    params_k_split = {"name": "gauss_rt", "var": var/2., "d": int(d)}
    split_kernel = partial(kernel_eval, params_k=params_k_split)
    swap_kernel = partial(kernel_eval, params_k=params_k_swap)
    delta = .5
    m = args.m
    n = int(2**(2*m))
    X = distribution.sample(n, rng_key)
    X = np.array(X)
    g = 0
    size = m
    halve_prob = delta / ( 4*(4**size)*(2**g)*( g + (2**g) * (size  - g) ) )
    thin_prob = delta * g / (g + ( (2**g)*(size - g) ))

    thin = partial(kt.thin, m=0, split_kernel = split_kernel, swap_kernel = swap_kernel, delta = thin_prob)
    halve = compress.symmetrize(lambda x: kt.thin(X = x, m=1, split_kernel = split_kernel, swap_kernel = swap_kernel, 
                                              unique=True, delta = halve_prob*(len(x)**2)))

    # coresets = kt.thin(X, m, split_kernel, swap_kernel, delta=delta, verbose=True)
    coresets = compress.compresspp(X, halve, thin, g)
    kt_samples = X[coresets, :]

    true_value = distribution.integral()
    iid_samples = distribution.sample(kt_samples.shape[0], rng_key)
    iid_estimate = mmd_flow.utils.evaluate_integral(distribution, iid_samples)
    iid_err = jnp.abs(true_value - iid_estimate)
    kt_estimate = mmd_flow.utils.evaluate_integral(distribution, kt_samples)
    kt_err = jnp.abs(true_value - kt_estimate)

    print(f'True value: {true_value}')
    print(f'IID err: {iid_err}')
    print(f'KT err: {kt_err}')
    jnp.save(f'{args.save_path}/kt_err.npy', kt_err)
    jnp.save(f'{args.save_path}/iid_err.npy', iid_err)
    jnp.save(f'{args.save_path}/iid_samples.npy', iid_samples)
    jnp.save(f'{args.save_path}/kt_samples.npy', kt_samples)
    return

if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print('Program started!')
    print(vars(args))
    main(args)
    print('Program finished!')
    new_save_path = args.save_path + '__complete'
    os.rename(args.save_path, new_save_path)
    print(f'Job completed. Renamed folder to: {new_save_path}')