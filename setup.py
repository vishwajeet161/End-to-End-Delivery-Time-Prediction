# In Python, setup.py is a module used to build and distribute Python package.
# It typically contains information about the package, such as its name, version, and dependencies as well as instruction for building and installing the package.
# It is used if we want to release it(project) as a package or library in future.

from setuptools import setup, find_packages
from typing import List

PROJECT_NAME = "Machine Learning"
VERSION = "0.0.1"
DESCRIPTION = "This is out machine learning project in modular coding"
AUTHOR_NAME = "Vishwajeet Patel"
AUTHOR_EMAIL = "vishwajeet.patel161@gmail.com"
REQUIREMENTS_FILE_NAME = "requirement.txt"

HYPHEN_E_DOT = "-e ."

# Requirement.txt file open
# read
# replace \n with empty string
def get_requirements_list()->List[str]:
    with open(REQUIREMENTS_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]

        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)

        return requirement_list


setup(name=PROJECT_NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR_NAME,
      author_email=AUTHOR_EMAIL,
    #   url='https://www.python.org/sigs/distutils-sig/',
      packages=find_packages(),
      install_requires = get_requirements_list()
     )