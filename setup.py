import os
from glob import glob

from setuptools import find_packages, setup


def write_version_file(version_: str) -> None:
    with open("src/tsipy/version.py", "w") as f:
        f.write('__version__ = "{}"\n'.format(version_))


version = "1.0.2"
write_version_file(version_=version)

setup(
    version=version,
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"tsipy": ["py.typed"]},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    ],
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
    project_urls={
        "Source": "https://github.com/roksikonja/tsipy",
        "Documentation": "https://tsipy.readthedocs.io/en/latest/",
    },
    extras_require={
        "dev": [
            "pytest",
            "black",
            "mypy",
            "isort",
            "flake8",
            "pylint",
            "jupyter",
        ],
        "docs": [
            "sphinx",
            "sphinxcontrib-bibtex",
            "sphinx-autodoc-typehints",
            "sphinx-rtd-theme",
            "sphinxcontrib-spelling",
            "ipython",
            "nbsphinx",
            "nbsphinx_link",
            "pandoc",
        ],
    },
)
