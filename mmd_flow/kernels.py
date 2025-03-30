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
        kme_RBF_Gaussian_vmap_func = jax.vmap(kme_RBF_Gaussian_func_)
        return kme_RBF_Gaussian_vmap_func(X)
    
    def mean_mean_embedding(self, mu1, mu2, Sigma1, Sigma2) -> float:
        return kme_double_RBF_Gaussian(mu1, mu2, Sigma1, Sigma2, self.sigma)

        
# class laplace_kernel(base_kernel):
#     sigma: float

#     def __call__(self, x: Array, y: Array) -> Array:
#         return jnp.exp(-jnp.sum(jnp.abs(_rescale(x - y, self.sigma))))

# class imq_kernel(base_kernel):
#     sigma: float
#     c: float = 1.0
#     beta: float = -0.5

#     def __call__(self, x: Array, y: Array) -> Array:
#         return jnp.power(
#             self.c**2 + _l2_norm_squared(_rescale(x - y, self.sigma)), self.beta
#         )


# class negative_distance_kernel(base_kernel):
#     sigma: float

#     def __call__(self, x: Array, y: Array) -> Array:
#         return -_l2_norm_squared(_rescale(x - y, self.sigma))


# class energy_kernel(base_kernel):
#     # x0: Array
#     beta: float
#     sigma: float
#     eps: float = 1e-8

#     def __call__(self, x: Array, y: Array) -> Array:
#         x0 = jnp.zeros_like(x)

#         pxx0 = jnp.power(_l2_norm_squared(_rescale(x - x0, self.sigma)) + self.eps, self.beta / 2)
#         pyx0 = jnp.power(_l2_norm_squared(_rescale(y - x0, self.sigma)) + self.eps, self.beta / 2)
#         pxy = jnp.power(_l2_norm_squared(_rescale(x - y, self.sigma)) + self.eps, self.beta / 2)

#         ret = 0.5 * (pxx0 + pyx0 - pxy)
#         return ret


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

# def kme_double_RBF_Gaussian(mu, Sigma, l):
#     """
#     The implementation of the initial of the RBF kernel with Gaussian distribution.

#     Args:
#         mu: Gaussian mean, (D, )
#         Sigma: Gaussian covariance, (D, D)
#         l: scalar

#     Returns:
#         initial error: scalar
#     """
#     l_ = l ** 2
#     D = mu.shape[0]
#     Lambda = jnp.eye(D) * l_
#     Lambda_inv = jnp.eye(D) / l_
#     part1 = jnp.linalg.det(jnp.eye(D) + Sigma @ Lambda_inv)
#     part2 = jnp.linalg.det(jnp.eye(D) + Sigma @ jnp.linalg.inv(Lambda + Sigma))
#     return part1 ** (-0.5) * part2 ** (-0.5)

