from setuptools import setup

setup(
    name='attitudeutils',
    version='0.1.0',
    author='Jeremie X. J. Bannwarth',
    author_email='jban039@aucklanduni.ac.nz',
    packages=['attitudeutils'],
    url='https://github.com/JBannwarth/attitudeutils',
    license='LICENSE',
    description='Functions to describe attitude and convert between attitude representations',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
    ],
)