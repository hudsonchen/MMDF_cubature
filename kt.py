import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import argparse
from mmd_flow.distributions import Distribution, Empirical_Distribution
from mmd_flow.kernels import gaussian_kernel
from mmd_flow.mmd import mmd_fixed_target
import mmd_flow.utils
from goodpoints.jax.kt import kernel_split, kernel_swap
from goodpoints.jax.rounding import log2_ceil
from goodpoints.jax.mmd import compute_mmd
import time
import pickle
from mmd_flow.typing import Array
from mmd_flow.kernels import kme_RBF_Gaussian_func
from functools import partial
from goodpoints.jax.sliceable_points import SliceablePoints
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def get_config():
    parser = argparse.ArgumentParser(description='mmd_flow_cubature')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--kernel', type=str, default='Gaussian')
    parser.add_argument('--step_size', type=float, default=0.1) # Step size will be rescaled by lmbda, the actual step size = step size * lmbda
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--bandwidth', type=float, default=0.1)
    parser.add_argument('--step_num', type=int, default=100, help='Number of KT-swap iterations')
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--inject_noise_scale', type=float, default=0.0)
    parser.add_argument('--integrand', type=str, default='neg_exp')
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"kt/{args.dataset}_dataset/{args.kernel}_kernel/"
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

class GaussianKernel:
    def __init__(self, sqd_bandwidth):
        '''A Gaussian kernel of the form exp(-.5*||x-y||^2/sqd_bandwidth)
        '''
        # Initialize exponential scale factor
        self.scale = -.5/sqd_bandwidth

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, points_x, points_y):
        x = points_x.get('p')
        y = points_y.get('p')
        return jnp.exp(((x - y) ** 2).sum(-1) * self.scale)

    def prepare_input(self, p):
        return SliceablePoints({'p': p})
    
class GaussianKernelMean0:
    def __init__(self, distribution):
        '''A Gaussian kernel of the form exp(-.5*||x-y||^2/sigma^2)
        shifted to be mean-zero with respect to the input distribution.

        Args:
            distribution: a Distribution object with methods mean_embedding(x), mean_mean_embedding(),
                and attribute kernel (a Gaussian kernel object with attribute sigma)
        '''
        self.distribution = distribution
        # Double expectation of the Gaussian kernel under distribution
        self.mean_mean_embedding = self.distribution.mean_mean_embedding()
        # Initialize exponential scale factor
        self.scale = -.5/distribution.kernel.sigma**2

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, points_x, points_y):
        x = points_x.get('p')
        y = points_y.get('p')
        # Mean-center kernel based on distribution
        val = (jnp.exp(((x - y) ** 2).sum(-1) * self.scale)
                - self.distribution.mean_embedding(x) 
                - self.distribution.mean_embedding(y) 
                + self.mean_mean_embedding)
        #print(f"success" )
        return val

    def prepare_input(self, p):
        return SliceablePoints({'p': p})

class gaussian_kernel_override(gaussian_kernel):
    # Override kernel mean embedding to handle (M,B,D) and (D,) inputs
    def mean_embedding(self, X: Array, mu: Array, Sigma: Array) -> Array:
        """
        The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution
        A fully vectorized implementation.

        Args:
            mu: Gaussian mean, (D, )
            Sigma: Gaussian covariance, (D, D)
            X: (M, D)

        Returns:
            kernel mean embedding: (M, )
        """
        kme_RBF_Gaussian_func_ = partial(kme_RBF_Gaussian_func, mu, Sigma, self.sigma)
        if X.ndim == 1:
            # Handle inputs of shape (D,)
            return kme_RBF_Gaussian_func_(X)
        kme_RBF_Gaussian_vmap_func = jax.vmap(kme_RBF_Gaussian_func_)
        if X.ndim == 3:
            # Add another vmap layer to handle (M, B, D) input
            return jax.vmap(kme_RBF_Gaussian_vmap_func)(X)
        return kme_RBF_Gaussian_vmap_func(X)
        
def main(args):
    # Generate multiple seeds from input seed
    rng_key = jax.random.PRNGKey(args.seed)
    kt_seed = args.seed
    particle_num = 2 ** int(args.m)

    kernel = gaussian_kernel_override(args.bandwidth)
    other_integrand = 'square' if args.integrand != 'square' else 'neg_exp'
    if args.dataset == 'gaussian':
        distribution = Distribution(kernel=kernel, means=jnp.array([[0.0, 0.0]]), covariances=jnp.eye(2)[None, :], 
                                    integrand_name=args.integrand, weights=None)
    elif args.dataset == 'mog':
        covariances = jnp.load('data/mog_covs.npy')
        means = jnp.load('data/mog_means.npy')
        k = 20
        weights = jnp.ones(k) / k
        distribution = Distribution(kernel=kernel, means=means, covariances=covariances, integrand_name=args.integrand, weights=weights)
    elif args.dataset == 'house_8L':
        data = np.genfromtxt('data/house_8L.csv', delimiter=',', skip_header=1)[:,:-1]
        distribution = Empirical_Distribution(kernel=kernel, samples=data, integrand_name=args.integrand)
    elif args.dataset == 'elevators':
        data = np.genfromtxt('data/elevators.csv', delimiter=',', skip_header=1)[:,:-1]
        distribution = Empirical_Distribution(kernel=kernel, samples=data, integrand_name=args.integrand)
    else:
        raise ValueError('Dataset not recognized!')

    # Use Gaussian kernel for KT-split
    split_kernel = GaussianKernel(kernel.sigma**2) 

    empirical = isinstance(distribution, Empirical_Distribution)
    if empirical:
        # Compute KT-split target input sample size
        N = distribution.samples.shape[0]
        target_n = min(N, particle_num**2)
        # Choose actual KT-split input sample size 
        # n_split = particle_num times the smallest power of 2 >= target_n/particle_num
        n_split = particle_num * (2**int(np.ceil(np.log2(target_n / particle_num))))
        # Sample indices into distribution.samples
        print(f'Sampling {n_split} indices from {N} data points')
        inds_key, _ = jax.random.split(rng_key)
        if n_split <= N:
            # Sample points without replacement
            inds = jax.random.choice(inds_key, N, shape=(n_split,), replace=False)
        else:
            # Include each point n//N times and sample remainder without replacement
            inds = jnp.concatenate(
                [jnp.arange(N).repeat(n_split//N), jax.random.choice(inds_key, N, shape=(n_split % N,), replace=False)], 
                axis=0)
        # Sort indices
        inds = jnp.sort(inds)
        X = distribution.samples[inds, :]
        # Prepare n_split input points for KT-split
        split_points = split_kernel.prepare_input(X)

        # Use Gaussian kernel for KT-swap
        swap_kernel = GaussianKernel(kernel.sigma**2)
        mean_zero = False
        baseline = True
        # Prepare all input points for KT-swap
        swap_points = swap_kernel.prepare_input(distribution.samples)

        # Precompute mean_mean_embedding in a memory-efficient way
        print(f'Precomputing mean_mean_embedding for MMD evaluation')
        start_time = time.time()
        distribution.mean_mean_embedding_val = compute_mmd(
            swap_kernel, swap_points, mode='mean-zero')
        distribution.mean_mean_embedding = lambda : distribution.mean_mean_embedding_val
        print(f'Elapsed: {time.time() - start_time}s')

        # Evaluate second integrand
        other_distribution = Empirical_Distribution(
            kernel=kernel, samples=distribution.samples, integrand_name=other_integrand)
    else:
        # Find the smallest power of 2 greater than or equal to particle_num
        nout_pow2 = 2**int(np.ceil(np.log2(particle_num)))
        # Thin down from sample size n = particle_num * nout_pow2 * 2^g
        n_split = particle_num * nout_pow2
        g = 2
        n = n_split * (2**g)
        X = distribution.sample(n, rng_key)
        # Prepare first n_split input points for KT-split
        split_points = split_kernel.prepare_input(X[:n_split])

        # Use mean-0 Gaussian kernel for KT-swap
        swap_kernel = GaussianKernelMean0(distribution)
        mean_zero = True
        baseline = False
        # Prepare all input points for KT-swap
        swap_points = swap_kernel.prepare_input(X)

        # Evaluate second integrand
        other_distribution = Distribution(
            kernel=kernel, means=distribution.means, covariances=distribution.covariances, 
            integrand_name=other_integrand, weights=distribution.weights)        
    divergence = mmd_fixed_target(args, kernel, distribution)

    #
    # KT-split
    #

    rng_gen = np.random.default_rng(kt_seed) # Random number generator for KT
    t = log2_ceil(n_split, particle_num) # Number of halving rounds
    delta = 0.5 # Failure probability parameter

    start_time = time.time()
    coresets = kernel_split(split_kernel, split_points, t, delta,
                            rng_gen)
    print(f'Split time: {time.time() - start_time}s')

    if empirical:
        # Coresets currently index into X; update to index into distribution.samples
        coresets = [inds[coreset] for coreset in coresets]

    #
    # KT-swap
    #
    start_time = time.time()
    num_repeat = args.step_num
    coreset = kernel_swap(
        swap_kernel, swap_points, 
        coresets, rng_gen, mean_zero=mean_zero, 
        num_repeat=num_repeat, random_swap_order=True, 
        inplace=True, baseline=baseline)
    print(f'Swap time: {time.time() - start_time}s')
    kt_samples = swap_points.get('p')[coreset, :]

    print(kt_samples.shape[0])
    
    # Evaluate single-function integration error for integrand
    true_value = distribution.integral()
    iid_samples = distribution.sample(kt_samples.shape[0], rng_key)
    iid_estimate = mmd_flow.utils.evaluate_integral(distribution, iid_samples)
    iid_err = jnp.abs(true_value - iid_estimate)
    kt_estimate = mmd_flow.utils.evaluate_integral(distribution, kt_samples)
    kt_err = jnp.abs(true_value - kt_estimate)

    # Evaluate MMD
    iid_mmd = divergence(iid_samples)
    kt_mmd = divergence(kt_samples)

    # Evaluate single-function integration error for the other integrand
    true_value_other = other_distribution.integral()
    iid_estimate_other = mmd_flow.utils.evaluate_integral(other_distribution, iid_samples)
    iid_err_other = jnp.abs(true_value_other - iid_estimate_other)
    kt_estimate_other = mmd_flow.utils.evaluate_integral(other_distribution, kt_samples)
    kt_err_other = jnp.abs(true_value_other - kt_estimate_other)

    print(f'True value {args.integrand}: {true_value}')
    print(f'IID err {args.integrand}: {iid_err}')
    print(f'KT err {args.integrand}: {kt_err}\n')
    print(f'True value {other_integrand}: {true_value_other}')
    print(f'IID err {other_integrand}: {iid_err_other}')
    print(f'KT err {other_integrand}: {kt_err_other}\n')
    print(f'IID MMD: {iid_mmd}')
    print(f'KT MMD: {kt_mmd}\n')
    jnp.save(f'{args.save_path}/iid_samples.npy', iid_samples)
    jnp.save(f'{args.save_path}/kt_samples.npy', kt_samples)
    jnp.save(f'{args.save_path}/kt_mmd.npy', kt_mmd)
    jnp.save(f'{args.save_path}/iid_mmd.npy', iid_mmd)
    jnp.save(f'{args.save_path}/kt_err_{other_integrand}.npy', kt_err_other)
    jnp.save(f'{args.save_path}/iid_err_{other_integrand}.npy', iid_err_other)
    jnp.save(f'{args.save_path}/kt_err_{args.integrand}.npy', kt_err)
    jnp.save(f'{args.save_path}/iid_err_{args.integrand}.npy', iid_err)
    # jnp.save(f'{args.save_path}/kt_err.npy', kt_err)
    # jnp.save(f'{args.save_path}/iid_err.npy', iid_err)
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