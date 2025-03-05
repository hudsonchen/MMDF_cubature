import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
import jax
import jax.numpy as jnp
import numpy as np
import sys
import pwd
import argparse
import scipy
from mmd_flow.distributions import Distribution
from mmd_flow.kernels import gaussian_kernel
from scipy.linalg import qr
from dppy.finite_dpps import FiniteDPP
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
    args.save_path += f"dpp/{args.dataset}_dataset/{args.kernel}_kernel/"
    args.save_path += f"__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__particle_num_{args.particle_num}__inject_noise_scale_{args.inject_noise_scale}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


def dpp_sampler(seed, mean, cov, N):
    rng = np.random.RandomState(seed)
    N = int(N)
    # Step 1: Define the number of samples we want and candidate points
    M = 3 * N  # Number of candidate points (should be > N)
    d = 2
    # Step 2: Generate M candidate points from a Gaussian distribution
    X = rng.multivariate_normal(np.zeros(d).flatten(), np.eye(d), size=M)  # Gaussian sample

    # Step 3: Compute Hermite polynomials at these candidate points
    Phi = np.zeros((M, N))  # M rows (candidates), N columns (features)

    for i in range(N):  # Limit to N features
        Phi[:, i] = scipy.special.eval_hermitenorm(i, X[:, 0]) * scipy.special.eval_hermitenorm(i, X[:, 1])

    # Step 4: Perform QR decomposition to get an orthonormal basis
    eig_vecs, _ = qr(Phi, mode='economic')  # Gives an MxN orthonormal matrix

    # Step 5: Construct Projection DPP
    DPP = FiniteDPP(kernel_type='correlation', projection=True,
                    **{'K_eig_dec': (np.ones(N), eig_vecs)})  # All eigenvalues = 1

    # Step 6: Sample exactly N points from the Projection DPP
    DPP.sample_exact(mode='GS', random_state=rng)

    # Get sampled indices and corresponding points
    selected_indices = DPP.list_of_samples[0]  # Get the first sampled subset
    dpp_samples = X[selected_indices, :]  # Extract exactly N sampled points

    L = np.linalg.cholesky(cov)
    dpp_samples = dpp_samples @ L.T + mean
    DPP.compute_K()
    dpp_weights = 1.0 / np.diag(DPP.K[selected_indices, :][:, selected_indices])
    return dpp_samples, dpp_weights


def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    N = args.particle_num
    d = 2
    kernel = gaussian_kernel(args.bandwidth)
    if args.dataset == 'gaussian':
        distribution = Distribution(kernel=kernel, means=jnp.array([[0.0, 0.0]]), covariances=jnp.eye(2)[None, :], 
                                    integrand_name=args.integrand, weights=None)
    elif args.dataset == 'mog':
        covariances = jnp.load('data/mog_covs.npy')
        means = jnp.load('data/mog_means.npy')
        k = 20
        weights = jnp.ones(k) / k
        distribution = Distribution(kernel=kernel, means=means, covariances=covariances, integrand_name=args.integrand, weights=weights)
    else:
        raise ValueError('Dataset not recognized!')
    
    if args.dataset == 'gaussian':
        dpp_samples, dpp_weights = dpp_sampler(args.seed, jnp.array([0.0, 0.0]), jnp.eye(2), N)
    elif args.dataset == 'mog':
        # Sample DPP points
        component_indices = jax.random.choice(rng_key, k, shape=(N,), p=weights)
        unique_components, sample_sizes = jnp.unique(component_indices, return_counts=True)

        mean = means[unique_components]
        cov = covariances[unique_components]

        # Generate samples for each unique Gaussian component
        samples_dict = {
            int(unique_components[i]): dpp_sampler(i + args.seed, mean[i], cov[i], sample_sizes[i])
            for i in range(len(unique_components))
        }
        dpp_samples = jnp.concatenate([samples_dict[int(idx)][0] for idx in samples_dict.keys()], axis=0)
        dpp_weights = jnp.concatenate([samples_dict[int(idx)][1] for idx in samples_dict.keys()], axis=0)

    iid_samples = distribution.sample(args.particle_num, rng_key)
    true_value = distribution.integral()
    iid_estimate = mmd_flow.utils.evaluate_integral(distribution, iid_samples)
    iid_err = jnp.abs(true_value - iid_estimate)
    
    # dpp_estimate = mmd_flow.utils.evaluate_integral(distribution, dpp_samples, dpp_weights)
    dpp_estimate = mmd_flow.utils.evaluate_integral(distribution, dpp_samples)
    dpp_err = jnp.abs(true_value - dpp_estimate)
    print(f'True value: {true_value}')
    print(f'IID err: {iid_err}')
    print(f'DPP err: {dpp_err}')
    jnp.save(f'{args.save_path}/dpp_err.npy', dpp_err)
    jnp.save(f'{args.save_path}/iid_err.npy', iid_err)
    jnp.save(f'{args.save_path}/iid_samples.npy', iid_samples)
    jnp.save(f'{args.save_path}/dpp_samples.npy', dpp_samples)
    return

if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print('Program started!')
    print(vars(args))
    main(args)
    print('Program finished!')