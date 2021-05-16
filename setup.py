import os
from glob import glob

from setuptools import find_packages, setup

# Load version file into version variable
with open(os.path.join("src/tsipy", "__version__.py")) as f:
    code = compile(f.read(), f.name, "exec")
    exec(code)

setup(
    version=version,  # noqa: F821
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"tsipy": ["py.typed"]},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    ],
    python_requires=">=3.7",
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
            "twine",
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
