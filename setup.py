from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as file:
    long_description = file.read()

with open('requirements.txt', 'r') as file:
    requires = file.read().splitlines()

setup(
    name='cnneuro-ds-generator',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='Dataset generator for the CNNeuro project',
    long_description=long_description,
    packages=find_packages(include=['cnneuro-ds-generator', 'cnneuro-ds-generator.*']),
    install_requires=requires
)
