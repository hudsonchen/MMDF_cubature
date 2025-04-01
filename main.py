import os
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
import jax
import jax.numpy as jnp
import numpy as np
import sys
import pwd
import argparse
from mmd_flow.distributions import Distribution
from mmd_flow.kernels import gaussian_kernel
from mmd_flow.mmd import mmd_fixed_target
from mmd_flow.gradient_flow import gradient_flow
import mmd_flow.utils
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
    parser.add_argument('--particle_num', type=int, default=20)
    parser.add_argument('--inject_noise_scale', type=float, default=0.0)
    parser.add_argument('--integrand', type=str, default='neg_exp')
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"mmd_flow/{args.dataset}_dataset/{args.kernel}_kernel/"
    args.save_path += f"__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__particle_num_{args.particle_num}__inject_noise_scale_{args.inject_noise_scale}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    N = args.particle_num
    d = 2
    kernel = gaussian_kernel(args.bandwidth)
    if args.dataset == 'gaussian':
        distribution = Distribution(kernel=kernel, means=jnp.array([[0.0, 0.0]]), covariances=jnp.eye(2)[None, :], 
                                    integrand_name=args.integrand, weights=None)
        Y = jax.random.normal(rng_key, shape=(N, d)) + 1. # initial particles
    elif args.dataset == 'mog':
        covariances = jnp.load('data/mog_covs.npy')
        means = jnp.load('data/mog_means.npy')
        k = 20
        weights = jnp.ones(k) / k
        distribution = Distribution(kernel=kernel, means=means, covariances=covariances, integrand_name=args.integrand, weights=weights)
        Y = jax.random.normal(rng_key, shape=(N, d)) / 10. + 0.0 # initial particles
    else:
        raise ValueError('Dataset not recognized!')
    
    divergence = mmd_fixed_target(args, kernel, distribution)
    info_dict, trajectory = gradient_flow(divergence, rng_key, Y, args)
    rate = int(args.step_num // 200)
    mmd_flow.utils.save_animation_2d(args, trajectory, kernel, distribution, rate, rng_key, args.save_path)
    
    true_value = distribution.integral()
    iid_samples = distribution.sample(args.particle_num, rng_key)
    iid_estimate = mmd_flow.utils.evaluate_integral(distribution, iid_samples)
    iid_err = jnp.abs(true_value - iid_estimate)

    mmd_flow_estimate = mmd_flow.utils.evaluate_integral(distribution, trajectory[-1, :, :])
    mmd_flow_err = jnp.abs(true_value - mmd_flow_estimate)

    print(f'True value: {true_value}')
    print(f'IID err: {iid_err}')
    print(f'MMD flow err: {mmd_flow_err}')
    jnp.save(f'{args.save_path}/mmd_flow_err.npy', mmd_flow_err)
    jnp.save(f'{args.save_path}/iid_err.npy', iid_err)
    jnp.save(f'{args.save_path}/iid_samples.npy', iid_samples)
    jnp.save(f'{args.save_path}/mmd_flow_samples.npy', trajectory[-1, :, :])

    return
    

if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print('Program started!')
    print(vars(args))
    main(args)
    print('Program finished!')