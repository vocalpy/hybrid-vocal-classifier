#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    version='1.0b',
    description='Python package for automated segmentation and classification of vocalizations',
    author='David Nicholson',
    author_email='nicholdav at gmail dot com',
    url='https://github.com/NickleDave/hybrid-vocal-classifier',
    packages=find_packages(),
    package_data={'': ['*.yml']},  # install with yml files e.g. from hvc.parse
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.5',
    ]
    )


