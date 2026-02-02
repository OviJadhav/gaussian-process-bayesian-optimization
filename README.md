# Gaussian Processes with Bayesian Optimization

A from-scratch implementation of Gaussian Processes and Bayesian Optimization for hyperparameter tuning.

### Synthetic Training Data
<img width="848" height="397" alt="image" src="https://github.com/user-attachments/assets/cf0931eb-08a9-4c14-8d8d-51e69171f9fd" />

### Fitting Gaussian Process with RBF Kernel + Visualizing Predictions with Uncertainity
<img width="849" height="362" alt="image" src="https://github.com/user-attachments/assets/6055de13-8ffd-4b7e-92de-454818aaef14" />

### Effects of Kernel Hyperparameters
<img width="848" height="311" alt="image" src="https://github.com/user-attachments/assets/7fb75d09-711a-4d4e-9c46-7a81906efe68" />

### Different Kernel Comparisons 
<img width="851" height="239" alt="image" src="https://github.com/user-attachments/assets/96a35f88-8265-42f8-a4b8-80bde5b6c033" />

### Sampling from GP Prior and Posterior
<img width="850" height="275" alt="image" src="https://github.com/user-attachments/assets/8135521a-8b8c-4d33-8e8e-3f8866bc49a9" />



## Project Structure
```
GP_and_BO/
├── src/                    # Core implementation
│   ├── gp_core/           # Gaussian Process
│   ├── bayesian_optimization/  # BO framework
│   └── utils/             # Utilities
├── notebooks/             # Interactive tutorials
├── data/                  # Datasets
├── experiments/           # Experiment scripts
└── tests/                 # Unit tests
```

## Getting Started
```bash
# virtual environment
python -m venv venv
source venv/bin/activate  

# dependencies
pip install -r requirements.txt

# quick test
python test_quick.py
```

## Author

Ovi Jadhav
- LinkedIn: [https://www.linkedin.com/in/ovijadhav/]
- GitHub: [https://github.com/OviJadhav]
