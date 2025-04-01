from setuptools import setup, find_packages

setup(
    name="mindxlib",
    version="0.1.0",
    author='Alibaba Damo Academy',
    author_email="TODO@alibaba-inc.com",
    description="Explainable AI methods from Alibaba Damo Academy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DAMO-DI-ML/mindxlib",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0", 
        "scikit-learn>=1.0.0",
        "shap>=0.41.0",
        "lime>=0.2.0",
        "pyroaring",
        "mip",
        "numba"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)