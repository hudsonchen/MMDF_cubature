import jax.numpy as jnp
import jax
import time
from functools import partial
from .typing import Array, Scalar, Distribution
from typing import Callable

class mmd_fixed_target:
    def __init__(self, args, kernel, distribution):
        self.kernel = kernel
        self.distribution = distribution
        self.args = args
    
    def get_witness_function(
        self, z, Y
    ) -> Scalar:
        z = z[None, :]
        K_zX = self.distribution.mean_embedding(z)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (-K_zX + K_zY.mean(1)).squeeze()

    def get_first_variation(self, Y) -> Callable:
        return partial(self.get_witness_function, Y=Y)
    
    def __call__(self, Y, rng_key):
        target_samples = self.distribution.sample(Y.shape[0], rng_key)
        K_XX = self.kernel.make_distance_matrix(target_samples, target_samples)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(target_samples, Y)
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()