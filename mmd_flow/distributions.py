import jax.numpy as jnp
import jax

class Distribution:
    def __init__(self, kernel, means, covariances, weights=None):
        """
        A class that supports Gaussian and Mixture of Gaussians distributions.

        Parameters:
        - kernel: the kernel
        - means: (d,) array for a single Gaussian mean, or (k, d) for MoG.
        - covariances: (d, d) for a single Gaussian, or (k, d, d) for MoG.
        - weights: (k,) array for MoG. If None, assumes a single Gaussian.
        """
        self.kernel = kernel
        self.means = jnp.atleast_2d(means)  # Ensure shape (k, d)
        self.covariances = jnp.atleast_3d(covariances)  # Ensure shape (k, d, d)
        self.k, self.d = self.means.shape

        if weights is None:
            self.weights = jnp.array([1.0])  # Single Gaussian case
        else:
            self.weights = jnp.asarray(weights)
            assert len(self.weights) == self.k, "Weights must match number of components."
            assert jnp.isclose(self.weights.sum(), 1), "Weights must sum to 1."

    def mean_embedding(self, Y):
        if self.k == 1:
            KME = self.kernel.mean_embedding(Y, self.means[0], self.covariances[0])
            return KME
        else:
            KME = jnp.zeros(len(Y))
            for i in range(self.k):
                KME += self.weights[i] * self.kernel.mean_embedding(Y, self.means[i], self.covariances[i])
        return KME
    
    def sample(self, sample_size, rng_key):
        """
        Sample from the mixture of Gaussians.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        rng_key, subkey = jax.random.split(rng_key)
        component_indices = jax.random.choice(subkey, self.k, shape=(sample_size,), p=self.weights)

        # Generate multivariate normal samples
        rng_key, subkey = jax.random.split(rng_key)
        means = self.means[component_indices]
        covs = self.covariances[component_indices]

        # Sample from multivariate normals
        samples = means + jax.random.multivariate_normal(subkey, jnp.zeros(self.d), covs, shape=(sample_size,))
        return samples
    
    def pdf(self, Y):
        """
        Compute the probability density function of the mixture of Gaussians.

        Parameters:
        - Y: (n, d) array of points to evaluate the PDF at.

        Returns:
        - pdf: (n,) array of PDF values.
        """
        pdf = jnp.zeros(len(Y))
        for i in range(self.k):
            pdf += self.weights[i] * jax.scipy.stats.multivariate_normal.pdf(Y, self.means[i], self.covariances[i])
        return pdf