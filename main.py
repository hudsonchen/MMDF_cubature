import os
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
import jax
import jax.numpy as jnp
import numpy as np
import sys
import pwd
import argparse
from mmd_flow.distributions import Distribution, Empirical_Distribution, Cross
from mmd_flow.kernels import gaussian_kernel, laplace_kernel, matern_32_kernel
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
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--kernel', type=str, default='Gaussian')
    parser.add_argument('--step_size', type=float, default=0.1) # Step size will be rescaled by lmbda, the actual step size = step size * lmbda
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--bandwidth', type=float, default=0.1)
    parser.add_argument('--step_num', type=int, default=100000)
    parser.add_argument('--particle_num', type=int, default=20)
    parser.add_argument('--inject_noise_scale', type=float, default=0.0)
    parser.add_argument('--integrand', type=str, default='neg_exp')
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"mmd_flow/{args.dataset}_dataset/{args.kernel}_kernel/"
    args.save_path += f"__step_size_{round(args.step_size, 8)}__dim_{args.d}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__particle_num_{args.particle_num}__inject_noise_scale_{args.inject_noise_scale}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    N = args.particle_num
    if args.kernel == 'Gaussian':
        kernel = gaussian_kernel(args.bandwidth)
    elif args.kernel == 'Laplace':
        kernel = laplace_kernel(args.bandwidth)
    elif args.kernel == 'Matern_32':
        kernel = matern_32_kernel(args.bandwidth)
    else:
        raise ValueError('Kernel not recognized!')
    
    if args.dataset == 'gaussian':
        distribution = Distribution(kernel=kernel, means=jnp.array([[0.0, 0.0]]), covariances=jnp.eye(2)[None, :], 
                                    integrand_name=args.integrand, weights=None)
        d = 2
        Y = jax.random.normal(rng_key, shape=(N, d)) + 1. # initial particles
    elif args.dataset == 'mog':
        if args.d == 2:
            covariances = jnp.load('data/mog_covs.npy')
            means = jnp.load('data/mog_means.npy')
            k = 20
            d = 2
        else:
            d = args.d
            k = 20
            rng_key, _ = jax.random.split(rng_key)
            means = jax.random.normal(rng_key, shape=(k, d))
            for _ in range(3):  # a few iterations to separate a bit
                for i in range(k):
                    diff = means[i] - means  # (k, d)
                    dist_sq = jnp.sum(diff**2, axis=1, keepdims=True) + 1e-4
                    repulsion = jnp.sum(diff / dist_sq, axis=0)  # repel from others
                    means = means.at[i].add(100. * repulsion / k)
            
            covariances = jnp.zeros((k, d, d))
            for i in range(k):
                rng_key, _ = jax.random.split(rng_key)
                A = jax.random.normal(rng_key, shape=(d, d))
                cov = jnp.dot(A, A.T) + d * jnp.eye(d)  # Ensures positive definiteness
                covariances = covariances.at[i, :, :].set(cov)
        weights = jnp.ones(k) / k
        distribution = Distribution(kernel=kernel, means=means, covariances=covariances, integrand_name=args.integrand, weights=weights)
        Y = jax.random.normal(rng_key, shape=(N, d)) / 10. + 0.0 # initial particles
    elif args.dataset == 'house_8L':
        data = np.genfromtxt('data/house_8L.csv', delimiter=',', skip_header=1)[:,:-1]
        d = data.shape[1]
        distribution = Empirical_Distribution(kernel=kernel, samples=data, integrand_name=args.integrand)
        Y = jax.random.normal(rng_key, shape=(N, d)) / 3. + 0.0 # initial particles
    elif args.dataset == 'elevators':
        data = np.genfromtxt('data/elevators.csv', delimiter=',', skip_header=1)[:,:-1]
        d = data.shape[1]
        distribution = Empirical_Distribution(kernel=kernel, samples=data, integrand_name=args.integrand)
        Y = jax.random.normal(rng_key, shape=(N, d)) / 3. + 0.0
    elif args.dataset == 'cross':
        k = 2
        w = 0.2
        h = 1.0
        skip = 1.5
        distribution = Cross(kernel=kernel, w=w, h=h, k=k, skip=skip)
        d = 2
        Y = jax.random.normal(rng_key, shape=(N, d)) / 10. + jnp.array([[-0.75, 0.75]]) # initial particles

    else:
        raise ValueError('Dataset not recognized!')
    
    divergence = mmd_fixed_target(args, kernel, distribution)
    if args.dataset == 'cross':
        save_trajectory = True
    else:
        save_trajectory = False

    if save_trajectory:
        info_dict, trajectory = gradient_flow(divergence, rng_key, Y, save_trajectory, args)
        mmd_flow_samples = trajectory[-1, :, :]
        jnp.save(f'{args.save_path}/Ys.npy', trajectory)
        # rate = int(args.step_num // 200)
        if d == 2 and args.dataset == 'cross':
            # Save the animation
            rate = 10
            mmd_flow.utils.save_animation_2d(args, trajectory, kernel, distribution, rate, rng_key, args.save_path)
        else:
            pass
    else:
        info_dict, mmd_flow_samples = gradient_flow(divergence, rng_key, Y, save_trajectory, args)

    
    true_value = distribution.integral()
    iid_samples = distribution.sample(args.particle_num, rng_key)
    iid_estimate = mmd_flow.utils.evaluate_integral(distribution, iid_samples)
    iid_err = jnp.abs(true_value - iid_estimate)

    mmd_flow_estimate = mmd_flow.utils.evaluate_integral(distribution, mmd_flow_samples)
    mmd_flow_err = jnp.abs(true_value - mmd_flow_estimate)

    print(f'True value: {true_value}')
    print(f'IID err: {iid_err}')
    print(f'MMD flow err: {mmd_flow_err}')
    jnp.save(f'{args.save_path}/mmd_flow_err.npy', mmd_flow_err)
    jnp.save(f'{args.save_path}/iid_err.npy', iid_err)
    jnp.save(f'{args.save_path}/iid_samples.npy', iid_samples)
    jnp.save(f'{args.save_path}/mmd_flow_samples.npy', mmd_flow_samples)

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