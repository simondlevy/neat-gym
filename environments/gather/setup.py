#!/usr/bin/env python3

'''
Python distutils setup file for gym-gather module.

Copyright (C) 2021 Simon D. Levy

MIT License
'''

from setuptools import setup

setup(
    name='gym_gather',
    version='0.1',
    install_requires=['gym', 'numpy', 'Box2D'],
    description='Gym environment for food-gathering robot from Stanley et al. 2009',
    packages=['gym_gather', 'gym_gather.envs', 'gym_gather.geometry'],
    author='Simon D. Levy',
    author_email='simon.d.levy@gmail.com',
    url='https://github.com/simondlevy/neat-gym/tree/master/environments/gather',
    license='MIT',
    platforms='Linux; Windows; OS X'
)
