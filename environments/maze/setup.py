#!/usr/bin/env python3

'''
Python distutils setup file for gym-nsmaze module.

Copyright (C) 2020 Simon D. Levy

MIT License
'''

from setuptools import setup

setup(
    name='gym_nsmaze',
    version='0.1',
    install_requires=['gym', 'numpy', 'Box2D'],
    description='Gym environment for Maze robot from Lehman & Stanley 2011',
    packages=['gym_nsmaze', 'gym_nsmaze.envs', 'gym_nsmaze.geometry'],
    author='Simon D. Levy',
    author_email='simon.d.levy@gmail.com',
    url='https://github.com/simondlevy/gym-nsmaze',
    license='MIT',
    platforms='Linux; Windows; OS X'
)
