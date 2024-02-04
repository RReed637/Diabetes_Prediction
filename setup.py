from setuptools import find_packages, setup
from typing import List
HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List(str):
    '''This Function Returns all requirements'''
    get_requirements=[]
    with open('file_path') as file_obj:
        requierments=file_obj.readlines()
        requierments=[req.replace ("\n","") for req in requirements]

        if HYPEN_E_DOT in requierments:
            requierments.remove(HYPEN_E_DOT)

    return requierments


setup(
    name = 'Diabetes Predictor',
    version='0.0.1',
    author='Ryan',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)