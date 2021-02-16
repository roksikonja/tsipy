# TSIPY

Python package for processing TSI signals.
 
Follows algorithms as described in 
[Iterative Correction of Sensor Degradation and a Bayesian Multi-Sensor Data Fusion Method](https://arxiv.org/abs/2009.03091).

## Bugs fixed

- Denormalization of y_out_std.

## Future Releases

- Noise standard deviation extraction.
- Degradation correction in white kernel.

## Installation
    
    # venv
    # python:3.8.7
    pip install numpy pandas matplotlib scipy tables cvxpy gpflow scikit-learn

    # Dockerfile - INCOMPLETE USE ABOVE
    docker build -t python:tsipy .
    
    FROM python:3.8.7
    
    MAINTAINER Rok Sikonja <sikonjarok@gmail.com>
    
    RUN pip install numpy pandas matplotlib scipy tables cvxpy gpflow scikit-learn

## Usage

Demos can be found in ```./scripts```.

    # Activate venv
    # add utils to PYTHONPATH
    # add tsipy to PYTHONPATH
    
    venv/Scripts\activate

    python scripts/demo_generator.py

    python -c "import os; import sys; print(os.getcwd()); print(os.listdir()); print(sys.path)"

## Cite

    @misc{kolar2020iterative,
          title={Iterative Correction of Sensor Degradation and a Bayesian Multi-Sensor Data Fusion Method}, 
          author={Luka Kolar and Rok Šikonja and Lenart Treven},
          year={2020},
          eprint={2009.03091},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }
