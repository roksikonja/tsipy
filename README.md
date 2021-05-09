<a href='https://tsipy.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/tsipy/badge/?version=latest' alt='Documentation Status' />
</a>
<a href='https://github.com/roksikonja/tsipy/actions/workflows/ci.yml'>
    <img src='https://github.com/roksikonja/tsipy/actions/workflows/ci.yml/badge.svg' alt='CI status' />
</a>

# TSIPY

Python package for processing TSI signals.

Full documentation is available [online](https://tsipy.readthedocs.io/).

## Installation

    pip install tsipy


## Installation for Developers

    # Clone repository
    git clone https://github.com/roksikonja/tsipy.git
    
    # Setup venv
    python -m venv venv
    source venv/bin/activate

    pip install --upgrade pip
    pip install wheel

    # For development
    pip install -e .[dev]

    # For documentation
    pip install -e .[docs]


## Usage

Demos can be found in ```./scripts```.

    python ./scripts/exp_virgo.py
    python ./scripts/exp_fusion.py

## References

References can be found in ```./references```.
