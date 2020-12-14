# TSIPY

Python package for processing TSI signals.
 
Follows algorithms as described in 
[Iterative Correction of Sensor Degradation and a Bayesian Multi-Sensor Data Fusion Method](https://arxiv.org/abs/2009.03091).

## Bugs fixed

- Denormalization of y_out_std.

## Future features

- MultiWhiteKernel 

## Installation
    
    # Python 3.7.4
    pip install numpy pandas matplotlib scipy tables cvxpy gpflow scikit-learn
    
    
## Code formatter
    
    pip install black
    
    # format code
    black . --exclude="./venv"
    
## Cite

    @misc{kolar2020iterative,
          title={Iterative Correction of Sensor Degradation and a Bayesian Multi-Sensor Data Fusion Method}, 
          author={Luka Kolar and Rok Šikonja and Lenart Treven},
          year={2020},
          eprint={2009.03091},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }