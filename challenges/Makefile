#
# Makefile for convenience
#
# Copyright (C) 2020 Simon D. Levy
#
# MIT License
#

install:
	sudo python3 setup.py install

gdemo:
	python3 neat_gym_challenges/envs/gather.py --sensors

gedit:
	vim neat_gym_challenges/envs/gather.py

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/ visuals/ runs/

commit:
	git commit -a

flake:
	flake8 setup.py
	flake8 demos/*.py
	flake8 neat_gym_challenges/__init__.py
	flake8 neat_gym_challenges/envs/*.py
