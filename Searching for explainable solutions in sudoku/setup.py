import io
import os

from setuptools import find_packages
from setuptools import setup

with io.open("README.md", "rt", encoding="utf8") as f:
    readme = f.read()

setup(
    name="CSP-Visualizer",
    version="1",
    url="https://github.com/sigurdurhelga/csp",
    maintainer="Sigurdurhelgason",
    maintainer_email="sigurdur@sigurdur.me",
    description="A visualization tool for a Sudoku Solver for the XAI-Ru team",
    long_description=readme,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["flask"],
    extras_require={"test": ["pytest"]},
)
