#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
    'keras',
    'theano',
    'agnez',
    'six',
    'git+https://github.com/AgnezIO/agnez' # not in pip
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='x',
    version='0.1.0',
    description="X",
    long_description=readme + '\n\n' + history,
    author="Eder Santana",
    author_email='edercsjr@gmail.com',
    url='https://github.com/edersantana/x',
    packages=[
        'x',
    ],
    package_dir={'x':
                 'x'},
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=False,
    keywords='x',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
