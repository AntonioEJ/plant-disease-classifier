"""Setup del paquete plant-disease-classifier."""
from setuptools import setup, find_packages

setup(
    name="plant-disease-classifier",
    version="1.0.0",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.12.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
    ],
)
