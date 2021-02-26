import pathlib

from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="tsipy",
    version="0.0.1",
    description="Python package for processing TSI signals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roksikonja/tsipy",
    author="Rok Šikonja",
    author_email="sikonjarok@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    keywords="degradation correction, data fusion, scientific computing",
    packages=["tsipy"],
    python_requires=">=3.6, <4",
    install_requires=[
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
    project_urls={
        "Bug Reports": "https://github.com/roksikonja/tsipy/issues",
        "Source": "https://github.com/roksikonja/tsipy/",
    },
)
