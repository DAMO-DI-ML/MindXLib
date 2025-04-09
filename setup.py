from gettext import install
from setuptools import setup, find_packages

version = 1.0

install_requires = [
    'pyroaring',
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
]

# install_requires=[
#             'joblib>=0.11',
#             'scikit-learn>=0.21.2',
#             'torch',
#             'torchvision',
#             'cvxpy>==1.1',
#             'Image',
#             'h5py<3.0.0',
#             'keras==2.3.1',
#             'matplotlib',
#             'numpy',
#             'pandas',
#             'scipy>=0.17',
#             'xport',
#             'scikit-image', 
#             'requests',
#             'xgboost==1.1.0', 	    
#             'bleach>=2.1.0',
#             'docutils>=0.13.1',
#             'Pygments',
#             'osqp',	    
#             'lime==0.1.1.37',
#             'shap==0.34.0',
#             'tqdm',
#             'kaggle',
#             'otoc @ git+https://github.com/IBM/otoc@main#egg=otoc'
# 	]

setup(name="mindxlib",
    version=1.0,
    author='Alibaba Damo Academy', 
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "shap>=0.41.0",
    "lime>=0.2.0",
    ],
    python_requires=">=3.8",
    # author="TODO",
    author_email="TODO@alibaba-inc.com]",
    description="Explainable AI methods from Alibaba Damo Academy and common libraries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="TODO:github",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)