from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

setup(
    name="lifelike",
    version="0.1",
    description="Codes and dataset for Lifelike Agility and Play",
    keywords="Robotics, Reinforcement Learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.5, <4",
    install_requires=[
        "gym",
        "numpy",
        "scipy",
        "wheel",
        "pybullet",
        "absl-py",
        "scipy",
        "cloudpickle",
        "pandas",
        "matplotlib",
        "tensorflow==1.15",
        "protobuf==3.20.0",
    ],
)

print(find_packages(where="src"))
