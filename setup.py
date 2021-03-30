from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    python_requires=">=3.6, <3.8",
    install_requires=[
        "sphinx",
        "pytest",
        "mypy",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "tables",
        "cvxpy",
        "gpflow",
        "tensorflow",
        "scikit-learn",
    ],
)
