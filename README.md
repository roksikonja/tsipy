# TSIPY

Python package for processing TSI signals.
Full documentation is available [online](https://tsipy.readthedocs.io/).

## Installation

Tested with ```3.7.x```. In Python ```3.8.x``` there is a conflict between dependencies ```cvxpy``` and ```tensorflow```
.

    # install from source
    git clone https://github.com/roksikonja/tsipy.git
    pip install -e .

    # install manually
    pip install numpy pandas matplotlib scipy tables cvxpy gpflow tensorflow scikit-learn
    pip install -e . --no-deps

    # install using conda
    conda create --name tsipy python=3.7
    conda activate tsipy
    pip install -e .

## Usage

Demos can be found in ```./scripts```.

    python ./scripts/demo_degradation.py
    python ./scripts/demo_fusion.py
