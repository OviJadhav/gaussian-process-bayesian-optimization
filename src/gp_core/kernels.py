"""
Kernel functions for Gaussian Processes.

A kernel (covariance function) defines the similarity between data points.
It determines how the GP generalizes from observed data.
"""

import numpy as np
from abc import ABC, abstractmethod


class Kernel(ABC):
    """Base class for all kernel functions."""
    
    @abstractmethod
    def __call__(self, X1, X2):
        """
        Compute the kernel matrix between X1 and X2.
        
        Args:
            X1: Array of shape (n1, d) - first set of points
            X2: Array of shape (n2, d) - second set of points
            
        Returns:
            K: Array of shape (n1, n2) - kernel matrix
        """
        pass
    
    @abstractmethod
    def gradient(self, X1, X2):
        """Compute gradients w.r.t. kernel hyperparameters."""
        pass


class RBFKernel(Kernel):
    """
    Radial Basis Function (RBF) / Squared Exponential Kernel.
    
    k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))
    
    where:
        σ² (variance) controls the vertical scale
        l (length_scale) controls how quickly correlation decays with distance
    
    Properties:
        - Infinitely differentiable (smooth functions)
        - Universal approximator
        - Most commonly used kernel
    """
    
    def __init__(self, length_scale=1.0, variance=1.0):
        """
        Args:
            length_scale: Controls how far you need to move in input space
                         before function values become uncorrelated
            variance: Controls the average distance of function from mean
        """
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1, X2):
        """
        Compute RBF kernel matrix.
        
        Args:
            X1: (n1, d) array
            X2: (n2, d) array
            
        Returns:
            K: (n1, n2) kernel matrix
        """
        # Compute squared Euclidean distances
        # ||x - x'||² = ||x||² + ||x'||² - 2*x·x'
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)  # (n1, 1)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)  # (n2, 1)
        
        # (n1, n2)
        sq_dists = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        
        # Ensure numerical stability (distances should be non-negative)
        sq_dists = np.maximum(sq_dists, 0)
        
        # Apply RBF formula
        K = self.variance * np.exp(-sq_dists / (2 * self.length_scale**2))
        
        return K
    
    def gradient(self, X1, X2):
        """Compute gradient w.r.t. length_scale and variance."""
        K = self(X1, X2)
        
        # Gradient w.r.t. length_scale
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        sq_dists = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        
        dK_dl = K * sq_dists / (self.length_scale**3)
        
        # Gradient w.r.t. variance
        dK_dv = K / self.variance
        
        return {'length_scale': dK_dl, 'variance': dK_dv}
    
    def __repr__(self):
        return f"RBFKernel(length_scale={self.length_scale:.3f}, variance={self.variance:.3f})"


class MaternKernel(Kernel):
    """
    Matern Kernel - generalization of RBF that allows for less smooth functions.
    
    k(x, x') = σ² * (1 + √(2ν) * r / l) * exp(-√(2ν) * r / l)
    
    where r = ||x - x'||
    
    Special cases:
        ν = 0.5: Equivalent to Exponential kernel (not differentiable)
        ν = 1.5: Once differentiable
        ν = 2.5: Twice differentiable
        ν → ∞: Converges to RBF kernel
    """
    
    def __init__(self, length_scale=1.0, variance=1.0, nu=1.5):
        """
        Args:
            length_scale: Length scale parameter
            variance: Output variance
            nu: Smoothness parameter (0.5, 1.5, 2.5 common)
        """
        self.length_scale = length_scale
        self.variance = variance
        self.nu = nu
        
        if nu not in [0.5, 1.5, 2.5]:
            raise NotImplementedError("Only nu in {0.5, 1.5, 2.5} implemented")
    
    def __call__(self, X1, X2):
        """Compute Matern kernel matrix."""
        # Compute Euclidean distances
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        sq_dists = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        
        if self.nu == 0.5:
            # Exponential kernel
            K = self.variance * np.exp(-dists / self.length_scale)
        
        elif self.nu == 1.5:
            # Once differentiable
            scaled_dist = np.sqrt(3) * dists / self.length_scale
            K = self.variance * (1 + scaled_dist) * np.exp(-scaled_dist)
        
        elif self.nu == 2.5:
            # Twice differentiable
            scaled_dist = np.sqrt(5) * dists / self.length_scale
            K = self.variance * (1 + scaled_dist + scaled_dist**2 / 3) * np.exp(-scaled_dist)
        
        return K
    
    def gradient(self, X1, X2):
        """Compute gradients (simplified implementation)."""
        # For now, return None - gradient-based optimization not critical
        return None
    
    def __repr__(self):
        return f"MaternKernel(length_scale={self.length_scale:.3f}, variance={self.variance:.3f}, nu={self.nu})"


class PeriodicKernel(Kernel):
    """
    Periodic Kernel - for modeling periodic functions.
    
    k(x, x') = σ² * exp(-2 * sin²(π * ||x - x'|| / p) / l²)
    
    where:
        p: period of the function
        l: length scale (controls smoothness)
    
    Useful for: seasonal data, oscillating phenomena
    """
    
    def __init__(self, period=1.0, length_scale=1.0, variance=1.0):
        """
        Args:
            period: Period of the periodic function
            length_scale: Controls smoothness
            variance: Output variance
        """
        self.period = period
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1, X2):
        """Compute Periodic kernel matrix."""
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        sq_dists = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        
        # Apply periodic transformation
        sin_term = np.sin(np.pi * dists / self.period)
        K = self.variance * np.exp(-2 * sin_term**2 / self.length_scale**2)
        
        return K
    
    def gradient(self, X1, X2):
        """Compute gradients."""
        return None
    
    def __repr__(self):
        return f"PeriodicKernel(period={self.period:.3f}, length_scale={self.length_scale:.3f})"


class LinearKernel(Kernel):
    """
    Linear Kernel - for linear relationships.
    
    k(x, x') = σ² * (x - c)ᵀ(x' - c)
    
    Useful for modeling linear trends.
    """
    
    def __init__(self, variance=1.0, offset=0.0):
        """
        Args:
            variance: Variance parameter
            offset: Offset parameter c
        """
        self.variance = variance
        self.offset = offset
    
    def __call__(self, X1, X2):
        """Compute Linear kernel matrix."""
        X1_centered = X1 - self.offset
        X2_centered = X2 - self.offset
        
        K = self.variance * (X1_centered @ X2_centered.T)
        return K
    
    def gradient(self, X1, X2):
        """Compute gradients."""
        return None
    
    def __repr__(self):
        return f"LinearKernel(variance={self.variance:.3f}, offset={self.offset:.3f})"


class KernelSum(Kernel):
    """Sum of two kernels - useful for combining structures."""
    
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    
    def __call__(self, X1, X2):
        return self.kernel1(X1, X2) + self.kernel2(X1, X2)
    
    def gradient(self, X1, X2):
        return None
    
    def __repr__(self):
        return f"({self.kernel1} + {self.kernel2})"


class KernelProduct(Kernel):
    """Product of two kernels."""
    
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    
    def __call__(self, X1, X2):
        return self.kernel1(X1, X2) * self.kernel2(X1, X2)
    
    def gradient(self, X1, X2):
        return None
    
    def __repr__(self):
        return f"({self.kernel1} * {self.kernel2})"