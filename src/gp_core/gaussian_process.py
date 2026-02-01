"""
Gaussian Process Regression implementation.

A Gaussian Process defines a distribution over functions:
    f(x) ~ GP(m(x), k(x, x'))

where:
    m(x): mean function (usually 0)
    k(x, x'): covariance/kernel function
"""

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from .kernels import RBFKernel


class GaussianProcess:
    """
    Gaussian Process Regression.
    
    Core algorithm:
    1. Training: Compute K = k(X, X) + σ²I and factorize
    2. Prediction: Use GP posterior formulas for mean and variance
    
    Posterior predictive distribution:
        p(f* | X*, X, y) = N(μ*, Σ*)
        
    where:
        μ* = k(X*, X) @ K⁻¹ @ y
        Σ* = k(X*, X*) - k(X*, X) @ K⁻¹ @ k(X, X*)
    """
    
    def __init__(self, kernel=None, noise=1e-5, mean_function=None):
        """
        Initialize Gaussian Process.
        
        Args:
            kernel: Covariance function (default: RBF)
            noise: Observation noise variance (σ²)
            mean_function: Mean function m(x) (default: zero mean)
        """
        self.kernel = kernel if kernel is not None else RBFKernel()
        self.noise = noise
        self.mean_function = mean_function if mean_function is not None else lambda x: 0.0
        
        # Training data (set during fit)
        self.X_train = None
        self.y_train = None
        self.L = None  # Cholesky decomposition of K
        self.alpha = None  # K⁻¹ @ y (for efficient prediction)
        
    def fit(self, X, y):
        """
        Fit the Gaussian Process to training data.
        
        This computes and factorizes the kernel matrix K for efficient prediction.
        
        Args:
            X: Training inputs, shape (n, d)
            y: Training outputs, shape (n,) or (n, 1)
            
        Mathematical details:
            1. Compute K = k(X, X) + σ²I
            2. Factorize K = LLᵀ (Cholesky decomposition)
            3. Solve Lα' = y, then L^Tα = α' to get α = K⁻¹y
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).flatten()
        
        self.X_train = X
        self.y_train = y
        
        # Compute kernel matrix with noise
        K = self.kernel(X, X)
        K += self.noise * np.eye(len(X))  # Add noise to diagonal
        
        # Cholesky decomposition: K = L @ L.T
        # This is numerically stable and O(n³)
        try:
            self.L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more noise (jitter) for numerical stability
            print("Warning: Cholesky failed, adding jitter")
            K += 1e-6 * np.eye(len(X))
            self.L = cholesky(K, lower=True)
        
        # Solve for alpha = K⁻¹ @ y
        # Step 1: Solve L @ alpha' = y
        # Step 2: Solve L.T @ alpha = alpha'
        self.alpha = cho_solve((self.L, True), y)
        
        return self
    
    def predict(self, X_test, return_std=False, return_cov=False):
        """
        Make predictions at test points.
        
        Args:
            X_test: Test inputs, shape (n_test, d)
            return_std: If True, return standard deviation
            return_cov: If True, return full covariance matrix
            
        Returns:
            mean: Predictive mean, shape (n_test,)
            std (optional): Predictive standard deviation, shape (n_test,)
            cov (optional): Predictive covariance, shape (n_test, n_test)
            
        Mathematical formulas:
            μ* = k(X*, X) @ K⁻¹ @ y = k(X*, X) @ α
            σ²* = k(X*, X*) - k(X*, X) @ K⁻¹ @ k(X, X*)
        """
        if self.X_train is None:
            raise RuntimeError("GP not fitted yet. Call fit() first.")
        
        X_test = np.atleast_2d(X_test)
        
        # Compute k(X_test, X_train)
        K_test_train = self.kernel(X_test, self.X_train)
        
        # Predictive mean: μ* = k(X*, X) @ α
        mean = K_test_train @ self.alpha
        
        if not (return_std or return_cov):
            return mean
        
        # For variance/covariance computation
        # Solve L @ v = k(X_test, X_train).T
        v = solve_triangular(self.L, K_test_train.T, lower=True)
        
        if return_cov:
            # Full predictive covariance
            # Σ* = k(X*, X*) - v.T @ v
            K_test_test = self.kernel(X_test, X_test)
            cov = K_test_test - v.T @ v
            return mean, cov
        
        if return_std:
            # Predictive variance (diagonal of covariance)
            K_test_diag = np.array([self.kernel(x.reshape(1, -1), x.reshape(1, -1))[0, 0] 
                                   for x in X_test])
            var = K_test_diag - np.sum(v**2, axis=0)
            
            # Ensure variance is non-negative (numerical errors can make it slightly negative)
            var = np.maximum(var, 0)
            std = np.sqrt(var)
            
            return mean, std
    
    def sample_y(self, X, n_samples=1, random_state=None):
        """
        Draw samples from the GP prior or posterior.
        
        Args:
            X: Points at which to sample, shape (n, d)
            n_samples: Number of function samples to draw
            random_state: Random seed
            
        Returns:
            samples: shape (n_samples, n)
        """
        rng = np.random.RandomState(random_state)
        X = np.atleast_2d(X)
        
        if self.X_train is None:
            # Sample from prior: f ~ N(0, K)
            K = self.kernel(X, X)
            K += 1e-8 * np.eye(len(X))  # Small jitter for numerical stability
            L = cholesky(K, lower=True)
            
            # Generate samples: f = L @ z, where z ~ N(0, I)
            z = rng.normal(size=(len(X), n_samples))
            samples = L @ z
            
            return samples.T
        else:
            # Sample from posterior
            mean, cov = self.predict(X, return_cov=True)
            cov += 1e-8 * np.eye(len(X))  # Jitter
            L = cholesky(cov, lower=True)
            
            z = rng.normal(size=(len(X), n_samples))
            samples = mean[:, None] + L @ z
            
            return samples.T
    
    def log_marginal_likelihood(self):
        """
        Compute the log marginal likelihood of the training data.
        
        This is useful for hyperparameter optimization.
        
        Formula:
            log p(y | X) = -½ yᵀK⁻¹y - ½ log|K| - (n/2) log(2π)
        
        Returns:
            lml: Log marginal likelihood
        """
        if self.X_train is None:
            raise RuntimeError("GP not fitted yet.")
        
        n = len(self.y_train)
        
        # -0.5 * y^T @ K^{-1} @ y = -0.5 * y^T @ alpha
        fit_term = -0.5 * self.y_train @ self.alpha
        
        # -0.5 * log|K| = -sum(log(diag(L)))
        log_det_term = -np.sum(np.log(np.diag(self.L)))
        
        # Constant term
        const_term = -0.5 * n * np.log(2 * np.pi)
        
        lml = fit_term + log_det_term + const_term
        
        return lml
    
    def optimize_hyperparameters(self, X, y, n_restarts=5):
        """
        Optimize kernel hyperparameters by maximizing log marginal likelihood.
        
        This is a simplified implementation using scipy.optimize.
        
        Args:
            X: Training inputs
            y: Training outputs
            n_restarts: Number of random restarts for optimization
        """
        from scipy.optimize import minimize
        
        def objective(params):
            """Negative log marginal likelihood."""
            # Update kernel parameters
            self.kernel.length_scale = np.exp(params[0])
            self.kernel.variance = np.exp(params[1])
            
            # Fit and compute likelihood
            self.fit(X, y)
            return -self.log_marginal_likelihood()
        
        best_params = None
        best_lml = np.inf
        
        for _ in range(n_restarts):
            # Random initialization
            init_params = np.random.randn(2)
            
            result = minimize(objective, init_params, method='L-BFGS-B')
            
            if result.fun < best_lml:
                best_lml = result.fun
                best_params = result.x
        
        # Set best parameters
        self.kernel.length_scale = np.exp(best_params[0])
        self.kernel.variance = np.exp(best_params[1])
        self.fit(X, y)
        
        return self
    
    def __repr__(self):
        fitted = "fitted" if self.X_train is not None else "not fitted"
        return f"GaussianProcess(kernel={self.kernel}, noise={self.noise:.1e}, {fitted})"