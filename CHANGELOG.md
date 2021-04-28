# Changelog

## Future Releases

- Degradation correction in white kernel.
- Python multiple versions with tox.

## v1.0.2 - 28. 4. 2021

- LocalGP and automatic results reproduction.
- Add CI/CD github actions.
  - black  
  - mypy
  - flake8
  - isort
  - pytest
  
- Changed dependency for solving QPs: from ```cvxpy``` to ```qpsolvers```.
  
- Added pytest tests.
- Partially added docstrings.
- Noise standard deviation extraction.
- Experiments:
  - SPM preliminary.
  - ACRIM1 and HF using LocalGP.
  
- Added `y_center` attribute to SignalGenerator for centering created signals.

## v1.0.1 - 30. 3. 2021

- Project restructuring.
  - Add documentation.
  - Getting started.
  - Documentation.
  - Tutorials.
  - References.
  
- Added LocalGP.
  
- Added pytest tests.
- Partially added docstrings.

## v0.1.0 - 11. 3. 2021

- Add LocalGP.
    - Fix normalization, test with generator data.
    - Add test for Virgo dataset.

## v0.0.1 - 26. 2. 2021

- Denormalization of y_out_std.
