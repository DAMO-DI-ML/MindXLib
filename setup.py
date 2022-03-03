from setuptools import setup, find_packages


required_pypi = [
    'pyroaring',
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
]

setup(name='MindXLib',
        version='1.0',
        description='It is a module for interpretable algorithms',
        author='Mind, Alibaba Group', 
        packages=find_packages(),
        requires=required_pypi)