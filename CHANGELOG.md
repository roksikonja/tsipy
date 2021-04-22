# Changelog

## Future Releases

- Noise standard deviation extraction.
- Degradation correction in white kernel.

- Add documentation.
  - Getting started.
  - Documentation.
  - Tutorials.
  - References.
- LocalGP and automatic results reproduction.
- More stable API.
- Add CI/CD github actions.
  - mypy
  - flake8
  - pytest
  - test coverage
- Python multiple versions with tox.
  
## v1.0.0 - 30. 3. 2021

- Project restructuring.
- Added mypy typing.
- Added LocalGP.
- Changed dependency for solving QPs: from ```cvxpy``` to ```qpsolvers```.
- Added github action for mypy.
  
- Added pytest tests.
- Partially added docstrings.

## v0.1.0 - 11. 3. 2021

- Add LocalGP.
    - Fix normalization, test with generator data.
    - Add test for Virgo dataset.

## v0.0.1 - 26. 2. 2021

- Denormalization of y_out_std.
