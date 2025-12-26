import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from .typing import Array


def _rescale(x: Array, scale: Array) -> Array:
    return x / scale

def _l2_norm_squared(x: Array) -> Array:
    return jnp.sum(jnp.square(x))

class gaussian_kernel():
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(- 0.5 * _l2_norm_squared(_rescale(x - y, self.sigma)))

    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        return vmap(vmap(type(self).__call__, (None, None, 0)), (None, 0, None))(
            self, X, Y
        )
    
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
        if X.ndim == 2:
            kme_RBF_Gaussian_vmap_func = jax.vmap(kme_RBF_Gaussian_func_)
            return kme_RBF_Gaussian_vmap_func(X)
        if X.ndim == 3:
            # Add another vmap layer to handle (M, B, D) input
            kme_RBF_Gaussian_vmap_func = jax.vmap(jax.vmap(kme_RBF_Gaussian_func_))
            return kme_RBF_Gaussian_vmap_func(X)
    
    def mean_mean_embedding(self, mu1, mu2, Sigma1, Sigma2) -> float:
        return kme_double_RBF_Gaussian(mu1, mu2, Sigma1, Sigma2, self.sigma)

    
    def mean_embedding_uniform(self, a: Array, b: Array, X: Array) -> Array:
        """
        The implementation of the kernel mean embedding of the RBF kernel with Uniform distribution
        A fully vectorized implementation.

        Args:
            a: (D,)
            b: (D,)
            l: float
            X: (M, D)

        Returns:
            kernel mean embedding: (M, )
        """
        return kme_RBF_uniform(a, b, self.sigma, X)
    
class laplace_kernel():
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-jnp.sum(jnp.abs(_rescale(x - y, self.sigma))))
    
    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        return vmap(vmap(type(self).__call__, (None, None, 0)), (None, 0, None))(
            self, X, Y
        )

class matern_32_kernel:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x: Array, y: Array) -> Array:
        r = jnp.sqrt(jnp.sum(jnp.square(_rescale(x - y, self.sigma))))
        sqrt3_r = jnp.sqrt(3.0) * r
        return (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)

    
    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        return vmap(vmap(type(self).__call__, (None, None, 0)), (None, 0, None))(
            self, X, Y
        )

@jax.jit
def kme_RBF_Gaussian_func(mu, Sigma, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution.
    Not vectorized.

    Args:
        mu: Gaussian mean, (D, )
        Sigma: Gaussian covariance, (D, D)
        y: (D, )
        l: float

    Returns:
        kernel mean embedding: scalar
    """
    D = mu.shape[0]
    l_ = l ** 2
    Lambda = jnp.eye(D) * l_
    Lambda_inv = jnp.eye(D) / l_
    part1 = jnp.linalg.det(jnp.eye(D) + Sigma @ Lambda_inv)
    part2 = jnp.exp(-0.5 * (mu - y).T @ jnp.linalg.inv(Lambda + Sigma) @ (mu - y))
    return part1 ** (-0.5) * part2


@jax.jit
def kme_double_RBF_Gaussian(mu_1, mu_2, Sigma_1, Sigma_2, l):
    """
    Computes the double integral a gaussian kernel with lengthscale l, with two different Gaussians.
    
    Args:
        mu_1, mu_2: (D,) 
        Sigma_1, Sigma_2: (D, D)
        l : scalar

    Returns:
        A scalar: the value of the integral.
    """
    D = mu_1.shape[0]
    l_ = l ** 2
    Lambda = jnp.eye(D) * l_
    sum_ = Sigma_1 + Sigma_2 + Lambda
    part_1 = jnp.sqrt(jnp.linalg.det(Lambda) / jnp.linalg.det(sum_))
    sum_inv = jnp.linalg.inv(sum_)
    # Compute exponent: - (1/2) * mu^T * (Σ1 + Σ2 + Lambda)⁻¹ * Γ⁻¹ * mu
    exp_term = -0.5 * ((mu_1 - mu_2).T @ sum_inv @ (mu_1 - mu_2))
    exp_value = jnp.exp(exp_term)
    result = part_1 * exp_value
    return result


def kme_RBF_uniform_func(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Uniform distribution.
    Not vectorized.

    Args:
        a: float (lower bound)
        b: float (upper bound)
        l: float
        y: float

    Returns:
        kernel mean embedding: scalar
    """
    part1 = jnp.sqrt(jnp.pi / 2) * l / (b - a)
    part2 = jax.scipy.special.erf((b - y) / (l * jnp.sqrt(2))) - jax.scipy.special.erf((a - y) / (l * jnp.sqrt(2)))
    return part1 * part2

def kme_RBF_uniform_func_dim(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Uniform distribution.
    Not vectorized.

    Args:
        a: (D,)
        b: (D,)
        l: float
        y: (D,)

    Returns:
        kernel mean embedding: scalar
    """
    kme_RBF_uniform_func_ = partial(kme_RBF_uniform_func, l=l)
    kme_RBF_uniform_vmap_func = jax.vmap(kme_RBF_uniform_func_)
    kme_all_d = kme_RBF_uniform_vmap_func(a=a, b=b, y=y)
    return jnp.prod(kme_all_d)

def kme_RBF_uniform(a, b, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution
    A fully vectorized implementation.

    Args:
        a: (D,)
        b: (D,)
        l: float
        y: (M, D)

    Returns:
        kernel mean embedding: (M, )
    """
    kme_RBF_uniform_func_ = partial(kme_RBF_uniform_func_dim, a=a, b=b, l=l)
    kme_RBF_uniform_vmap_func = jax.vmap(kme_RBF_uniform_func_)
    kme_all_d = kme_RBF_uniform_vmap_func(y=y)
    return kme_all_d