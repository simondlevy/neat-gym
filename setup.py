#!/usr/bin/env python3

'''
Python distutils setup file for neat-gym package.

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from setuptools import setup

setup(
    name='neat_gym',
    version='0.1',
    install_requires=['gym', 'numpy'],
    description='Use NEAT to learn OpenAI Gym environment',
    packages=['neat_gym', 'neat_gym.novelty'],
    author='Simon D. Levy',
    author_email='simon.d.levy@gmail.com',
    url='https://github.com/simondlevy/gym-copter',
    license='MIT',
    platforms='Linux; Windows; OS X'
    )
