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
	vim neat-evolve.py

nedit:
	vim neat_gym/novelty/__init__.py

pole:
	neat-evolve.py config/cartpole

hyper:
	neat-evolve.py --hyper config/cartpole

es:
	neat-evolve.py --eshyper config/cartpole

plot:
	./neat-plot.py runs/CartPole-v1.csv

help:
	./neat-evolve.py --help

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/ visuals/ runs/

commit:
	git commit -a

flake:
	flake8 *.py
	flake8 neat_gym/__init__.py
	flake8 neat_gym/novelty/__init__.py
