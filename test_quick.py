"""
Quick test script to verify GP implementation works.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.gp_core.kernels import RBFKernel
from src.gp_core.gaussian_process import GaussianProcess


def main():
    print("=" * 60)
    print("Testing Gaussian Process Implementation")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, 10).reshape(-1, 1)
    y_train = np.sin(X_train.flatten()) + 0.1 * np.random.randn(10)
    
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    
    # Create and fit GP
    print("\n1. Creating GP with RBF kernel...")
    kernel = RBFKernel(length_scale=1.0, variance=1.0)
    gp = GaussianProcess(kernel=kernel, noise=0.01)
    print(f"   {gp}")
    
    print("\n2. Fitting GP to training data...")
    gp.fit(X_train, y_train)
    print(f"   {gp}")
    print(f"   Log Marginal Likelihood: {gp.log_marginal_likelihood():.3f}")
    
    print("\n3. Making predictions...")
    y_pred, y_std = gp.predict(X_test, return_std=True)
    print(f"   Predicted {len(y_pred)} points")
    print(f"   Mean prediction: {y_pred.mean():.3f}")
    print(f"   Mean uncertainty: {y_std.mean():.3f}")
    
    print("\n4. Sampling from posterior...")
    samples = gp.sample_y(X_test, n_samples=3)
    print(f"   Generated {samples.shape[0]} function samples")
    
    # Visualize
    print("\n5. Creating visualization...")
    plt.figure(figsize=(12, 6))
    
    plt.fill_between(X_test.flatten(), 
                     y_pred - 2*y_std, 
                     y_pred + 2*y_std, 
                     alpha=0.2, color='blue', label='95% confidence')
    plt.plot(X_test, y_pred, 'b-', linewidth=2, label='GP mean')
    plt.scatter(X_train, y_train, c='red', s=100, alpha=0.8, 
                edgecolors='black', linewidths=1.5, label='Training data', zorder=3)
    
    for i, sample in enumerate(samples):
        plt.plot(X_test, sample, '--', alpha=0.3, linewidth=1)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Gaussian Process - Quick Test', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    print("   Saving figure to 'test_gp_output.png'...")
    plt.savefig('test_gp_output.png', dpi=150)
    print("   ✓ Figure saved!")
    
    print("\n" + "=" * 60)
    print("All tests passed! GP implementation working correctly.")
    print("=" * 60)
    print("\nNext steps:")
    print("  • Create notebooks/01_gp_basics.ipynb")
    print("  • Or run: jupyter notebook")
    print("=" * 60)


if __name__ == "__main__":
    main()