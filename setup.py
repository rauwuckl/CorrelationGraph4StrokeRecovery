import os
from setuptools import setup, find_packages

PYPI_REQUIREMENTS = []
if os.path.exists('requirements.txt'):
    for line in open('requirements.txt'):
        PYPI_REQUIREMENTS.append(line.strip())


setup(
    name="StrokeRecovery", 
    packages=find_packages(),
    version='1.0.0',
    install_requires=PYPI_REQUIREMENTS,
    )