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
        description='Alibaba Damo explainabel AI',
        author='Mind, Alibaba Damo', 
        packages=find_packages(),
        include_package_data=True,
        install_requires=install_requires)