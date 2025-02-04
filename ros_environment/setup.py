from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['car_mpc_control'],
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'casadi>=3.5.5',
        'matplotlib>=3.1.0',
        'scikit-learn>=0.24.0',
    ]
)

setup(**d) 