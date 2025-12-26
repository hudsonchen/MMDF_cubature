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
from goodpoints.jax.compress import kt_compresspp
from goodpoints.jax.sliceable_points import SliceablePoints
import time
import pickle
import matplotlib.pyplot as plt
from mmd_flow.typing import Array
from mmd_flow.kernels import kme_RBF_Gaussian_func

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

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
    parser.add_argument('--seed', type=int, default=None)
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
    args.save_path += f"kt_plus/{args.dataset}_dataset/{args.kernel}_kernel/"
    args.save_path += f"__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__particle_num_{2 ** int(args.m)}__inject_noise_scale_{args.inject_noise_scale}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args
    
# We define a new Gaussian kernel here, because the way broadcasting is done in 
# goodpoints.jax.compress is different than our Gaussian kernel.
@partial(jax.jit, static_argnames=['distribution'])
def centered_gaussian_kernel(points_x, points_y, l, distribution, kme_kme):
    x, y = points_x.get("p"), points_y.get("p")
    k_xy = jnp.exp(-0.5 * jnp.sum((x - y) ** 2, axis=-1) / (l ** 2))
    kme_x = distribution.mean_embedding(x)
    kme_y = distribution.mean_embedding(y)
    return k_xy - kme_x - kme_y + kme_kme

@partial(jax.jit, static_argnames=['distribution'])
def uncentered_gaussian_kernel(points_x, points_y, l, distribution, kme_kme):
    x, y = points_x.get("p"), points_y.get("p")
    k_xy = jnp.exp(-0.5 * jnp.sum((x - y) ** 2, axis=-1) / (l ** 2))
    return k_xy
    
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
    if args.kernel == 'Gaussian':
        # centered kernel
        kernel_fn = partial(centered_gaussian_kernel, l=float(args.bandwidth), distribution=distribution, 
                            kme_kme=distribution.mean_mean_embedding())
        # uncentered kernel
        # kernel_fn = partial(uncentered_gaussian_kernel, l=float(args.bandwidth), distribution=distribution, 
                            # kme_kme=distribution.mean_mean_embedding())
    else:
        raise ValueError('Kernel not recognized!')

    m = args.m
    g = 2
    n = int(2**(2*m)) * (2 ** g)
    X = distribution.sample(n, rng_key)
    
    rng = np.random.default_rng(args.seed)
    points = SliceablePoints({"p": X}) 
    coresets = kt_compresspp(kernel_fn, points, w=np.ones(X.shape[0]) / X.shape[0], 
                             rng_gen=rng, inflate_size=n, g=g)
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
    import shutil
    if os.path.exists(new_save_path):
        shutil.rmtree(new_save_path)  # Deletes existing folder
    os.rename(args.save_path, new_save_path)
    print(f'Job completed. Renamed folder to: {new_save_path}')