#
# Makefile for convenience
#
# Copyright (C) 2020 Simon D. Levy
#
# MIT License
#

install:
	sudo python3 setup.py install

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

gdemo:
	python3 demos/gather-concentric.py --steps 100 --sensors

commit:
	git commit -a

flake:
	flake8 setup.py
	flake8 neat_gym_tests/__init__.py
	flake8 neat_gym_tests/envs/*.py
	flake8 demos/*.py
