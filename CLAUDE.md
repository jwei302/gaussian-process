# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project comparing different approaches for function approximation and extrapolation. The project studies a specific physics function related to phase transitions and critical temperature, comparing Gaussian Processes, Polynomial Regression, and Neural Networks.

## Environment Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

## Core Mathematical Function

The project centers around approximating this physics function:
```python
def f(T, m, Tc=1.25):
    return 0.5*(1-Tc/T)*m**2 + (Tc/T)**3*m**4/12
```

Where:
- T: Temperature parameter
- m: Magnetization parameter  
- Tc: Critical temperature (default 1.25)

## Notebook Structure and Execution Order

### main.ipynb
Primary comparative analysis notebook. Contains:
- Complete implementation of all three methods (GP, Polynomial, Neural Network)
- Domain splitting methodology (trains on high T, tests on low T extrapolation)
- Comprehensive visualizations and error analysis
- Theoretical bounds computation

### gp.ipynb  
Gaussian Process scaling experiments:
- Studies GP performance with varying training set sizes (100-2500 samples)
- Uses iterative kernel search algorithm for optimal kernel selection
- Focuses on extrapolation performance

### nn.ipynb
Large-scale neural network experiments:
- Uses 1M training samples for robust neural network training
- 3-layer architecture with 64 neurons per layer
- LBFGS optimizer for precise function approximation

### torch.ipynb
PyTorch experiments:
- FashionMNIST dataset loading and basic tensor operations
- Separate from main function approximation work

### remez.ipynb
Polynomial approximation with theoretical bounds:
- Implements both classical bounds and Remez exchange bounds
- Comparative analysis of bound tightness

## Key Algorithms

### Iterative Kernel Search
Automated kernel selection for Gaussian Processes:
```python
def iterative_kernel_search(X, y, base_kernels=base_kernels, max_iter=10):
    # Evaluates base kernels (Linear, RBF, Matern, RationalQuadratic)
    # Iteratively combines best kernels using + and * operations
    # Uses BIC for model selection
```

### Domain Rectangle Construction
```python
def build_rectangles(T, m):
    # Splits domain: training on high T, testing on low T
    # Returns R_full, R_train, R_test rectangles
```

### BIC Computation
```python
def compute_bic(gp, X, y):
    # Bayesian Information Criterion for GP model selection
    # Balances fit quality with model complexity
```

## Key Parameters

- `N = 10000`: Standard dataset size for main experiments
- `Tc = 1.25`: Critical temperature parameter
- `alpha = 1e-5`: Regularization parameter for Ridge regression
- Domain ranges: T ∈ [0.8, 2.2], m ∈ [-1.25, 1.25]
- Training domain: T ∈ [1.5, 2.2] (high temperature)
- Testing domain: T ∈ [0.8, 1.5] (low temperature extrapolation)

## Expected Outputs

Each notebook generates:
- Performance metrics (MAE, MSE) for train/test sets
- Comparison plots showing model predictions vs ground truth
- 3D surface visualizations of learned functions
- Theoretical bound comparisons (remez.ipynb)
- Training curves and convergence analysis

The main finding: Gaussian Processes achieve near-perfect training fit (MSE ~1e-15) and good extrapolation, while neural networks show overfitting, and polynomial regression provides interpretable results with theoretical guarantees.

## Data Dependencies

- FashionMNIST dataset (auto-downloaded to `data/FashionMNIST/`)
- All other data generated synthetically from the physics function

## Visualization Outputs

Figures saved to `figs/`:
- `surfaces.png`: 3D surface comparisons
- `mse_comparison.png`: Training vs testing MSE
- `theoretical_bound.png`: Bounds analysis
- `extrapolation.png`: Extrapolation performance