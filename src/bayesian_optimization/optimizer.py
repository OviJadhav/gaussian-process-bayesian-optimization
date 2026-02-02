"""
Bayesian Optimization implementation.

The main optimization loop that uses GP + acquisition functions to
efficiently find the optimum of expensive black-box functions.
"""

import numpy as np
from scipy.optimize import minimize
from ..gp_core.gaussian_process import GaussianProcess
from ..gp_core.kernels import RBFKernel
from .acquisition import ExpectedImprovement


class BayesianOptimizer:
    """
    Bayesian Optimization for black-box function optimization.
    
    Algorithm:
    1. Initialize with random samples
    2. Fit GP to observed data
    3. Use acquisition function to find next point
    4. Evaluate function at new point
    5. Repeat until budget exhausted
    
    Use when:
        - Function evaluations are expensive
        - No gradient information available
        - Want to minimize number of evaluations
        - Need global optimization (not just local)
    """
    
    def __init__(self, 
                 bounds,
                 gp=None,
                 acquisition_function=None,
                 n_initial=5,
                 n_restarts=10,
                 random_state=None):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            bounds: List of (min, max) tuples for each dimension
                   e.g., [(0, 1), (0, 1)] for 2D optimization in [0,1]²
            gp: Gaussian Process (default: GP with RBF kernel)
            acquisition_function: Acquisition function (default: EI)
            n_initial: Number of random initial samples
            n_restarts: Number of restarts for acquisition optimization
            random_state: Random seed for reproducibility
        """
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.n_initial = n_initial
        self.n_restarts = n_restarts
        self.rng = np.random.RandomState(random_state)
        
        # Default GP with RBF kernel
        self.gp = gp if gp is not None else GaussianProcess(
            kernel=RBFKernel(length_scale=1.0, variance=1.0),
            noise=1e-6
        )
        
        # Default acquisition function
        self.acquisition_function = acquisition_function if acquisition_function is not None \
                                   else ExpectedImprovement(xi=0.01)
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
    
    def _normalize(self, X):
        """Normalizing inputs to [0, 1] for GP stability."""
        X = np.atleast_2d(X)
        return (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
    
    def _denormalize(self, X_norm):
        """Denormalizing from [0, 1] back to original space."""
        X_norm = np.atleast_2d(X_norm)
        return X_norm * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]    


    def _random_sample(self, n_samples=1):
        """Generate random samples within bounds."""
        samples = self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_samples, self.n_dims)
        )
        return samples
    
    def _optimize_acquisition(self):
        """
        Find the point that maximizes the acquisition function.
        
        Uses multi-start gradient-based optimization.
        
        Returns:
            X_next: Next point to evaluate, shape (1, n_dims)
        """
        # Get current best observation
        y_best = np.max(self.y_observed)
        
        # Negative acquisition (for minimization)
        def neg_acquisition(X):
            X = X.reshape(1, -1)
            return -self.acquisition_function(X, self.gp, y_best)[0]
        
        # Multi-start optimization
        best_acq = np.inf
        best_x = None
        
        # Try multiple random starting points
        for _ in range(self.n_restarts):
            x0 = self._random_sample(1)[0]
            
            result = minimize(
                neg_acquisition,
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return best_x.reshape(1, -1)
    
    def suggest(self):
        """Suggest the next point to evaluate."""
        n_observed = len(self.y_observed)
        
        # Initial random exploration
        if n_observed < self.n_initial:
            return self._random_sample(1)
        
        # Normalize observed data for GP
        X_norm = self._normalize(np.array(self.X_observed))
        y = np.array(self.y_observed)
        
        # Fit GP to normalized data
        self.gp.fit(X_norm, y)
        
        # Optimize GP hyperparameters every 5 iterations
        if n_observed % 5 == 0:
            try:
                self.gp.optimize_hyperparameters(X_norm, y, n_restarts=3)
            except:
                pass  # If optimization fails, keep current hyperparameters
        
        # Optimize acquisition function in normalized space
        X_next_norm = self._optimize_acquisition_normalized()
        
        # Denormalize back to original space
        X_next = self._denormalize(X_next_norm)
        
        return X_next
    

    def _optimize_acquisition_normalized(self):
        """Optimizing acquisition in normalized [0,1] space."""
        y_best = np.max(self.y_observed)
        
        def neg_acquisition(X_norm):
            X_norm = X_norm.reshape(1, -1)
            return -self.acquisition_function(X_norm, self.gp, y_best)[0]
        
        # Bounds in normalized space
        norm_bounds = [(0, 1)] * self.n_dims
        
        best_acq = np.inf
        best_x = None
        
        for _ in range(self.n_restarts):
            x0 = self.rng.uniform(0, 1, self.n_dims)
            
            result = minimize(
                neg_acquisition,
                x0,
                bounds=norm_bounds,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return best_x.reshape(1, -1)
    

    def observe(self, X, y):
        """
        Record an observation.
        
        Args:
            X: Input point, shape (1, n_dims) or (n_dims,)
            y: Function value at X (scalar)
        """
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            self.X_observed.append(X[0])
            self.y_observed.append(float(y))
        else:
            # Multiple observations
            for x_i, y_i in zip(X, y):
                self.X_observed.append(x_i)
                self.y_observed.append(float(y_i))
    
    def optimize(self, objective_function, n_iterations=50, verbose=True):
        """
        Run Bayesian Optimization.
        
        Args:
            objective_function: Function to maximize
                               Should take array of shape (n_dims,) and return scalar
            n_iterations: Number of optimization iterations
            verbose: Print progress
            
        Returns:
            result: Dictionary with:
                - 'X_best': Best point found
                - 'y_best': Best value found
                - 'X_observed': All evaluated points
                - 'y_observed': All observed values
                - 'convergence': Convergence history (best value at each iteration)
        """
        convergence = []
        
        if verbose:
            print("=" * 60)
            print("Starting Bayesian Optimization")
            print(f"Bounds: {self.bounds.tolist()}")
            print(f"Iterations: {n_iterations}")
            print(f"Acquisition: {self.acquisition_function}")
            print("=" * 60)
        
        for i in range(n_iterations):
            # Suggest next point
            X_next = self.suggest()
            
            # Evaluate objective
            y_next = objective_function(X_next[0])
            
            # Record observation
            self.observe(X_next, y_next)
            
            # Track best value
            y_best_current = np.max(self.y_observed)
            convergence.append(y_best_current)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{n_iterations}: Best value = {y_best_current:.4f}")
        
        # Get best result
        best_idx = np.argmax(self.y_observed)
        X_best = self.X_observed[best_idx]
        y_best = self.y_observed[best_idx]
        
        if verbose:
            print("=" * 60)
            print(f"Optimization complete!")
            print(f"Best value: {y_best:.4f}")
            print(f"Best point: {X_best}")
            print("=" * 60)
        
        return {
            'X_best': X_best,
            'y_best': y_best,
            'X_observed': np.array(self.X_observed),
            'y_observed': np.array(self.y_observed),
            'convergence': np.array(convergence)
        }
    
    def __repr__(self):
        n_obs = len(self.y_observed)
        return f"BayesianOptimizer(dims={self.n_dims}, observations={n_obs}, acq={self.acquisition_function})"


# Example usage
if __name__ == "__main__":
    # Simple 1D test
    def test_function(x):
        """Simple 1D function with known optimum."""
        return -(x - 2)**2 + 1
    
    optimizer = BayesianOptimizer(
        bounds=[(0, 5)],
        n_initial=3,
        random_state=42
    )
    
    result = optimizer.optimize(test_function, n_iterations=20, verbose=True)
    
    print(f"\nTrue optimum: x=2, f(x)=1")
    print(f"Found optimum: x={result['X_best'][0]:.3f}, f(x)={result['y_best']:.3f}")