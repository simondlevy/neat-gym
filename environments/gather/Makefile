#
# Makefile for convenience
#
# Copyright (C) 2021 Simon D. Levy
#
# MIT License
#

install:
	sudo python3 setup.py install

edit:
	vim gym_gather/envs/__init__.py

clean:
	sudo rm -rf build/ dist/ *.egg-info __pycache__ */__pycache__ models/ visuals/ runs/

demo:
	python3 demos/concentric.py --steps 100 --sensors

commit:
	git commit -a

flake:
	flake8 setup.py
	flake8 gym_gather/envs/__init__.py
