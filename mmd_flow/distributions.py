import jax.numpy as jnp
import jax
import scipy

class Distribution:
    def __init__(self, kernel, means, covariances, integrand_name, weights=None):
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
        self.integrand_name = integrand_name
        if integrand_name == 'square':
            self.integrand = lambda x: (x**2).sum(1)
        elif integrand_name == 'neg_exp':
            self.integrand = lambda x: jnp.exp(-(x**2).sum(1))
        else:
            raise ValueError('Function not recognized!')
        
        if weights is None:
            self.weights = jnp.array([1.0])  # Single Gaussian case
        else:
            self.weights = jnp.asarray(weights)
            assert len(self.weights) == self.k, "Weights must match number of components."
            assert jnp.isclose(self.weights.sum(), 1), "Weights must sum to 1."

    def mean_embedding(self, Y):
        # kme = jnp.zeros(len(Y))
        # for i in range(self.k):
        #     kme += self.weights[i] * self.kernel.mean_embedding(Y, self.means[i], self.covariances[i])
        # Vectorized computation using vmap
        kme_values = jax.vmap(self.kernel.mean_embedding, in_axes=(None, 0, 0))(Y, self.means, self.covariances)
        kme = jnp.tensordot(self.weights, kme_values, axes=1)
        return kme
    
    def mean_mean_embedding(self):
        if self.k == 1:
            double_kme = self.kernel.mean_mean_embedding(self.means[0], self.covariances[0])
            return double_kme
        else:
            double_kme = 0
            for i in range(self.k):
                for j in range(self.k):
                    double_kme += self.weights[i] * self.weights[j] * self.kernel.mean_mean_embedding(self.means[i], self.means[j], self.covariances[i], self.covariances[j])
            return double_kme
    
    def sample(self, sample_size, rng_key):
        """
        Sample i.i.d from the mixture of Gaussians.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        rng_key, _ = jax.random.split(rng_key)
        component_indices = jax.random.choice(rng_key, self.k, shape=(sample_size,), p=self.weights)

        means = self.means[component_indices, :]
        covs = self.covariances[component_indices, :, :]

        def sample_gaussian(mean, cov, key):
            return jax.random.multivariate_normal(key, mean, cov)

        subkeys = jax.random.split(rng_key, sample_size)
        samples = jax.vmap(sample_gaussian)(means, covs, subkeys)
        return samples
    
    def qmc_sample(self, sample_size, rng_key):
        """
        Sample QMC from the mixture of Gaussians.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        component_indices = jax.random.choice(rng_key, self.k, shape=(sample_size,), p=self.weights)
        unique_components, sample_sizes = jnp.unique(component_indices, return_counts=True)

        mean = self.means[unique_components]
        cov = self.covariances[unique_components]

        def generate_qmc_samples(mean, cov, size):
            sobol = scipy.stats.qmc.Sobol(self.d)
            u = jnp.array(sobol.random(size))  # Generate Sobol sequence
            L = jnp.linalg.cholesky(cov)      # Compute Cholesky decomposition
            return mean + jax.scipy.stats.norm.ppf(u) @ L.T

        # Generate samples for each unique Gaussian component
        samples_dict = {
            int(unique_components[i]): generate_qmc_samples(mean[i], cov[i], sample_sizes[i])
            for i in range(len(unique_components))
        }
        qmc_samples = jnp.concatenate([samples_dict[int(idx)] for idx in samples_dict.keys()], axis=0)
        return qmc_samples
    
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
    
    def integral(self):
        if self.integrand_name == 'square':
            integral = 0
            for i in range(self.k):
                integral += self.weights[i] * (jnp.trace(self.covariances[i]) + jnp.linalg.norm(self.means[i])**2)
        elif self.integrand_name == 'neg_exp':
            integral = 0
            for i in range(self.k):
                cov_inv = jnp.linalg.inv(self.covariances[i])
                temp = jnp.exp(0.5 * (self.means[i].T @ cov_inv @ jnp.linalg.inv(cov_inv + 2 * jnp.eye(self.d)) @ cov_inv @ self.means[i]))
                temp *= jnp.exp(-0.5 * self.means[i].T @ cov_inv @ self.means[i])
                temp * jnp.sqrt(jnp.linalg.det(2 * self.covariances[i] + jnp.eye(self.d)))
                cov_new = jnp.linalg.inv(cov_inv + 2 * jnp.eye(self.d))
                integral += self.weights[i] * temp * jnp.sqrt(jnp.linalg.det(cov_inv)) * jnp.sqrt(jnp.linalg.det(cov_new))
        return integral
    

class Empirical_Distribution:
    def __init__(self, kernel, samples, integrand_name):
        self.kernel = kernel
        self.samples = samples
        self.integrand_name = integrand_name
        self.n = len(samples)
        self.d = samples.shape[1]
        
        if integrand_name == 'square':
            self.integrand = lambda x: (x**2).sum(1)
        elif integrand_name == 'neg_exp':
            self.integrand = lambda x: jnp.exp(-(x**2).sum(1))
        else:
            raise ValueError('Function not recognized!')
        
    def mean_embedding(self, Y):
        """
        Compute the kernel mean embedding.

        Parameters:
        - Y: (n, d) array of points to evaluate 

        Returns:
        - pdf: (n,) array of PDF values.
        """
        kme = self.kernel.make_distance_matrix(Y, self.samples).mean(1)
        return kme
    
    def mean_mean_embedding(self):
        """
        Compute the kernel mean embedding of the empirical distribution.

        Returns:
        - double_kme: scalar, the value of the double integral.
        """
        double_kme = self.kernel.make_distance_matrix(self.samples, self.samples).mean()
        return double_kme
    
    def sample(self, sample_size, rng_key):
        """
        Sample i.i.d from the empirical distribution.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        rng_key, _ = jax.random.split(rng_key)
        indices = jax.random.choice(rng_key, self.n, shape=(sample_size,), replace=True)
        return self.samples[indices]
    
    def integral(self):
        """
        Compute the integral of the empirical distribution.

        Returns:
        - integral: scalar, the value of the integral.
        """
        if self.integrand_name == 'square':
            integral = (self.samples**2).sum(1).mean()
        elif self.integrand_name == 'neg_exp':
            integral = jnp.exp(-(self.samples**2).sum(1)).mean()
        return integral