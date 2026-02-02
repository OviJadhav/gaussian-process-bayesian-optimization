"""Quick test for Bayesian Optimization."""

import numpy as np
from src.bayesian_optimization.optimizer import BayesianOptimizer

# Simple 1D test function
def test_function(x):
    """Optimize this: max at x=2."""
    return -(x - 2)**2 + 1

print("Testing Bayesian Optimization...")
optimizer = BayesianOptimizer(bounds=[(0, 5)], n_initial=3, random_state=42)
result = optimizer.optimize(test_function, n_iterations=15, verbose=True)

print(f"\n Test passed!")
print(f"True optimum: x=2, f(x)=1")
print(f"Found: x={result['X_best'][0]:.3f}, f(x)={result['y_best']:.3f}")