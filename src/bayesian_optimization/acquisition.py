"""
Acquisition functions for Bayesian Optimization.

Acquisition functions determine where to sample next by balancing:
- Exploration: Sampling in uncertain regions
- Exploitation: Sampling near current best points
"""

import numpy as np
from scipy.stats import norm


class AcquisitionFunction:
    """Base class for acquisition functions."""
    
    def __call__(self, X, gp, y_best, **kwargs):
        """
        Compute acquisition function values.
        
        Args:
            X: Candidate points, shape (n, d)
            gp: Fitted Gaussian Process
            y_best: Current best observed value
            
        Returns:
            values: Acquisition values, shape (n,)
        """
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.
    
    EI(x) = E[max(f(x) - f(x_best), 0)]
          = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)
    
    where:
        Z = (μ(x) - f_best - ξ) / σ(x)
        Φ = CDF of standard normal
        φ = PDF of standard normal
        ξ = exploration parameter (usually small, e.g., 0.01)
    
    Properties:
        - Balances exploration (high σ) and exploitation (high μ)
        - Differentiable (can use gradient-based optimization)
        - Most popular acquisition function
    """
    
    def __init__(self, xi=0.01):
        """
        Args:
            xi: Exploration-exploitation trade-off parameter
                - xi = 0: Pure exploitation (greedy)
                - xi > 0: Encourages exploration
        """
        self.xi = xi
    
    def __call__(self, X, gp, y_best, **kwargs):
        """
        Compute Expected Improvement.
        
        Args:
            X: Candidate points, shape (n, d)
            gp: Fitted Gaussian Process
            y_best: Best observed value so far (for maximization)
            
        Returns:
            ei: Expected improvement values, shape (n,)
        """
        # Get GP predictions
        mu, sigma = gp.predict(X, return_std=True)
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        # Compute improvement
        improvement = mu - y_best - self.xi
        
        # Compute Z score
        Z = improvement / sigma
        
        # Expected Improvement formula
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei
    
    def __repr__(self):
        return f"ExpectedImprovement(xi={self.xi})"


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB(x) = μ(x) + κ * σ(x)
    
    where:
        μ(x) = predictive mean
        σ(x) = predictive standard deviation
        κ = exploration parameter
    
    Properties:
        - Simple and intuitive
        - κ controls exploration-exploitation trade-off
        - No need for current best value
        - Theoretical guarantees (no-regret bounds)
    """
    
    def __init__(self, kappa=2.0):
        """
        Args:
            kappa: Exploration parameter
                - kappa = 0: Pure exploitation (greedy)
                - kappa = 1-2: Balanced
                - kappa > 2: Heavy exploration
        """
        self.kappa = kappa
    
    def __call__(self, X, gp, y_best=None, **kwargs):
        """
        Compute Upper Confidence Bound.
        
        Args:
            X: Candidate points, shape (n, d)
            gp: Fitted Gaussian Process
            y_best: Not used (included for API consistency)
            
        Returns:
            ucb: UCB values, shape (n,)
        """
        # Get GP predictions
        mu, sigma = gp.predict(X, return_std=True)
        
        # UCB formula
        ucb = mu + self.kappa * sigma
        
        return ucb
    
    def __repr__(self):
        return f"UpperConfidenceBound(kappa={self.kappa})"


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement (PI) acquisition function.
    
    PI(x) = P(f(x) > f_best + ξ)
          = Φ((μ(x) - f_best - ξ) / σ(x))
    
    where:
        Φ = CDF of standard normal
        ξ = exploration parameter
    
    Properties:
        - Intuitive interpretation (probability of being better)
        - Tends to be more exploitative than EI
        - Less popular in practice
    """
    
    def __init__(self, xi=0.01):
        """
        Args:
            xi: Minimum improvement threshold
        """
        self.xi = xi
    
    def __call__(self, X, gp, y_best, **kwargs):
        """
        Compute Probability of Improvement.
        
        Args:
            X: Candidate points, shape (n, d)
            gp: Fitted Gaussian Process
            y_best: Best observed value so far
            
        Returns:
            pi: Probability of improvement values, shape (n,)
        """
        # Get GP predictions
        mu, sigma = gp.predict(X, return_std=True)
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)
        
        # Compute Z score
        Z = (mu - y_best - self.xi) / sigma
        
        # Probability of Improvement
        pi = norm.cdf(Z)
        
        return pi
    
    def __repr__(self):
        return f"ProbabilityOfImprovement(xi={self.xi})"


# Example usage and testing
if __name__ == "__main__":
    print("Acquisition Functions Implemented:")
    print(f"  • {ExpectedImprovement()}")
    print(f"  • {UpperConfidenceBound()}")
    print(f"  • {ProbabilityOfImprovement()}")
    print("\n All acquisition functions ready!")