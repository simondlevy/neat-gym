#
# Makefile for convenience
#
# Copyright (C) 2020 Simon D. Levy
#
# MIT License
#

install:
	sudo python3 setup.py install

edit:
	vim neat_gym_tests/envs/__init__.py

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/ visuals/ runs/

mdemo:
	python3 demos/maze-medium.py

hdemo:
	python3 demos/maze-hard.py

medium:
	neat-evolve.py config/medium --novelty 

hard:
	neat-evolve.py config/hard --novelty 

commit:
	git commit -a

flake:
	flake8 setup.py
	flake8 neat_gym_tests/envs/__init__.py
	flake8 demos/*.py
