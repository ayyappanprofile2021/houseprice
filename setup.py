# This file helps to build our entire application as a package.

# setuptools is a python library used for packaging and distributing
# Python projects. It helps you to creat the installable packages that
# can be installed via "pip".

# find_packages is a helper function in setuptools that automatically
# finds all python packages within your project so you dont have to 
# list them maually. 

from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of required packages
    mentioned in requirements.txt.
    '''
    requirements = []
    with open(file_path) as fileobj:
        requirements = fileobj.readlines()
        requirements = [req.replace("\n", " ") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements
setup(
    name='housepriceproject',
    version='0.0.1',
    author='Ayyappan',
    author_email='ayyappanprofile@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)





