import os
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
import jax
import jax.numpy as jnp
import numpy as np
import sys
import pwd
import scipy
import argparse
from mmd_flow.distributions import Distribution, Empirical_Distribution
from mmd_flow.kernels import gaussian_kernel
from mmd_flow.mmd import mmd_fixed_target
from mmd_flow.gradient_flow import gradient_flow
import mmd_flow.utils
from tqdm import tqdm
import time
import pickle
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


def get_config():
    parser = argparse.ArgumentParser(description='mmd_flow_cubature')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
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
    args.save_path += f"support_points/{args.dataset}_dataset/{args.kernel}_kernel/"
    args.save_path += f"__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__particle_num_{args.particle_num}__inject_noise_scale_{args.inject_noise_scale}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


@jax.jit
def cdist(x, y):
    return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


def loss(Y, X):
    n = X.shape[0]
    N = Y.shape[0]

    dists_x_y = cdist(X, Y)
    dists_x_x = cdist(X, X)
    dists_y_y = cdist(Y, Y)
    term1 = (2 / n / N) * jnp.sum(dists_x_y)
    term2 = (1 / n**2) * jnp.sum(dists_x_x)
    term3 = (1 / N**2) * jnp.sum(dists_y_y)
    return -term1 + term2 + term3

@jax.jit
def update_support(Y, X):
    n, d = X.shape
    N = Y.shape[0]

    diff = X[:, None, :] - X[None, :, :]  # shape (n, n, d)
    norm = jnp.linalg.norm(diff, axis=-1, keepdims=True) + 1e-10

    mask = ~jnp.eye(n, dtype=bool)
    masked_diff = (diff / norm) * mask[..., None]
    term1 = (N / n) * np.sum(masked_diff, axis=1)  # shape (n, d)

    dists = cdist(X, Y) + 1e-10
    weights = jnp.ones(Y.shape[0])
    term2 = jnp.dot(dists**-1, (weights[:, None] * Y))  # shape (n, d)
    q = jnp.dot(dists**-1, weights)  # shape (n,)

    M = (term1 + term2) / q[:, None]
    return M, q

def fit_mm(distribution, X, maxit, tol, verbosity, wgts, rng_key):
    dbar = jnp.zeros(X.shape[0])
    for itr in tqdm(range(maxit)):
        rng_key, _ = jax.random.split(rng_key)
        Y = distribution.sample(X.shape[0], rng_key=rng_key)
        w = wgts[itr]
        M, q = update_support(Y, X)
        kappa = w * q / (w * q + (1-w) * dbar)
        X_next = (1 - kappa[:, None]) * X + kappa[:, None] * M
        dbar = w * q + (1-w) * dbar
        # dis = jnp.linalg.norm(X_next - X)
        # if verbosity > 1:
        #     print(f"{itr:5d} {dis:12.5f} {loss(Y, X):12.5f}")
        # if dis < tol:
        #     success = True
        #     break
        X = X_next
    return X


def get_start(Y, n, rng_key):
    N, d = Y.shape
    idx = jax.random.permutation(rng_key, N)[:n]
    return Y[idx, :] + 0.1 * jax.random.normal(rng_key, shape=(n, d))

def support_points(args, distribution, npt, tol_mm, verbosity, rng_key):
    X = get_start(distribution.sample(args.particle_num, rng_key), npt, rng_key)
    n, d = X.shape
    wgts = (n*d) / (n*d + jnp.arange(args.step_num + 1))
    X = fit_mm(distribution, X, maxit=args.step_num, tol=tol_mm, verbosity=verbosity, wgts=wgts, rng_key=rng_key)
    return X


def main(args):
    rng_key = jax.random.PRNGKey(args.seed)
    N = args.particle_num
    kernel = gaussian_kernel(args.bandwidth)
    if args.dataset == 'gaussian':
        d = 2
        distribution = Distribution(kernel=kernel, means=jnp.array([[0.0, 0.0]]), covariances=jnp.eye(2)[None, :], 
                                    integrand_name=args.integrand, weights=None)
    elif args.dataset == 'mog':
        covariances = jnp.load('data/mog_covs.npy')
        means = jnp.load('data/mog_means.npy')
        k = 20
        weights = jnp.ones(k) / k
        d = 2
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

    true_value = distribution.integral()
    iid_samples = distribution.sample(args.particle_num, rng_key)
    iid_estimate = mmd_flow.utils.evaluate_integral(distribution, iid_samples)
    iid_err = jnp.abs(true_value - iid_estimate)
    sp_samples = support_points(args, distribution, npt=args.particle_num, tol_mm=1e-4, verbosity=0, rng_key=rng_key)
    sp_estimate = mmd_flow.utils.evaluate_integral(distribution, sp_samples)
    sp_err = jnp.abs(true_value - sp_estimate)

    print(f'True value: {true_value}')
    print(f'IID err: {iid_err}')
    print(f'Support Points err: {sp_err}')
    jnp.save(f'{args.save_path}/sp_err.npy', sp_err)
    jnp.save(f'{args.save_path}/iid_err.npy', iid_err)
    jnp.save(f'{args.save_path}/iid_samples.npy', iid_samples)
    jnp.save(f'{args.save_path}/sp_samples.npy', sp_samples)

    return

if __name__ == "__main__":
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