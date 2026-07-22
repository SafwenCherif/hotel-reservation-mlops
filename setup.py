from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Hotel-Reservation1",
    version="0.2",
    author="Safwen",
    packages=find_packages(),
    install_requires = requirements,
)