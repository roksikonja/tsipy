from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"tsipy": ["py.typed"]},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    python_requires=">=3.7, <3.9",
    install_requires=[
        "numpy",
        "tensorflow",
        "gpflow",
        "qpsolvers",
        "pandas",
        "matplotlib",
        "scipy",
        "tables",
        "scikit-learn",
    ],
)
