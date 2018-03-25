"""iris setup."""
from setuptools import setup

setup(
    name='iris',
    version='0.1.2',
    description='A python-based wavefront sensing module',
    long_description='',
    license='Copyright (C) 2017-2018 Brandon Dube, all rights reserved',
    author='Brandon Dube',
    author_email='brandondube@gmail.com',
    packages=['iris'],
    install_requires=['numpy', 'matplotlib', 'scipy', 'prysm', 'pandas', 'pyyaml'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ]
)
