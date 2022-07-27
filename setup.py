from setuptools import setup, find_packages

setup(
    name='wam-envs',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),    
    install_requires=['gym>=0.15.4, <0.16.0'],
)