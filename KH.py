import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import sys
import pwd
import argparse
from mmd_flow.distributions import Distribution
from mmd_flow.kernels import gaussian_kernel
from mmd_flow.mmd import mmd_fixed_target
from mmd_flow.gradient_flow import gradient_flow
import mmd_flow.utils
import matplotlib.pyplot as plt
import time
import pickle
from tqdm import tqdm
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--kernel', type=str, default='Gaussian')
    parser.add_argument('--step_size', type=float, default=0.1) # Step size will be rescaled by lmbda, the actual step size = step size * lmbda
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--bandwidth', type=float, default=0.1)
    parser.add_argument('--step_num', type=int, default=10000)
    parser.add_argument('--particle_num', type=int, default=20)
    parser.add_argument('--integrand', type=str, default='neg_exp')
    args = parser.parse_args()  
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"kernel_herding/{args.dataset}_dataset/{args.kernel}_kernel/"
    args.save_path += f"__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__particle_num_{args.particle_num}"
    args.save_path += f"__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


def herd(distribution, totalSS, rng_key):
    #-- calculate totalSS super samples from the distribution estimated by samples with kernel hyperparam gamma
    numDim = distribution.d
    #init vars
    xss = np.zeros((totalSS,numDim)) #open space in mem for array of super samples
    #gradient descent can have some probems, so make bounds to terminate if goes too far away
    minBound, maxBound = -5, 5
    bestSeed = jax.random.normal(rng_key, shape=[numDim,])  # Adjust mean and std as needed
    for i in tqdm(range(1, totalSS)):
        f = lambda x: -distribution.mean_embedding(x[None, :])[0] + 1./(i+1) * (distribution.kernel.make_distance_matrix(xss[:i,:], x[None, :]).sum())
        results = scipy.optimize.minimize(f, bestSeed, method='L-BFGS-B', bounds=[(minBound, maxBound)])
        #if grad descent failed, pick a random sample and try again
        if jnp.min(results.x) < minBound or jnp.max(results.x) > maxBound:
            bestSeed = jax.random.normal(rng_key, shape=results.x.shape)  # Adjust mean and std as needed
            print("Gradient descent failed..............")
            continue
    
        #grad descent succeeded (yay!), so assign new value to super samples
        xss[i,:]=results.x

        if distribution.d == 2:
            plt.figure()
            resolution = 100
            x_vals = jnp.linspace(minBound, maxBound, resolution)
            y_vals = jnp.linspace(minBound, maxBound, resolution)
            X, Y = jnp.meshgrid(x_vals, y_vals)
            grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)
            logpdf = jnp.log(distribution.pdf(grid).reshape(resolution, resolution))
            contour = plt.contourf(X, Y, logpdf, levels=20, cmap='viridis')
            plt.colorbar(contour, label="Log PDF")
            plt.scatter(xss[:i+1,0], xss[:i+1,1], color='red', marker='x', s=10)
            plt.savefig(f'{args.save_path}/kernel_herd_{i+1}.png')
            plt.close()
    return xss

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
        Y = jax.random.normal(rng_key, shape=(N, d)) + 0.5 # initial particles
    else:
        raise ValueError('Dataset not recognized!')
    
    SS = herd(distribution, args.particle_num, rng_key)
    jnp.save(f'{args.save_path}/kernel_herding_samples.npy', SS)
    kh_err, iid_err = mmd_flow.utils.evaluate_integral(args, distribution, SS, rng_key)
    jnp.save(f'{args.save_path}/kh_err.npy', kh_err)
    jnp.save(f'{args.save_path}/iid_err.npy', iid_err)
    return
    

if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    print('Program started!')
    print(vars(args))
    main(args)
    print('Program finished!')