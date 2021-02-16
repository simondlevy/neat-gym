#!/usr/bin/env python3

'''
Python distutils setup file for NEAT-Gym test environments.

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from setuptools import setup

setup(
    name='neat_gym_tests',
    version='0.1',
    install_requires=['gym', 'numpy', 'Box2D'],
    description='Gym environments for NEAT',
    packages=['neat_gym_tests',
              'neat_gym_tests.envs',
              'neat_gym_tests.geometry'],
    author='Simon D. Levy',
    author_email='simon.d.levy@gmail.com',
    url='https://github.com/simondlevy/neat-gym',
    license='MIT',
    platforms='Linux; Windows; OS X'
)
