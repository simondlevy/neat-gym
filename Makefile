install:
	sudo python3 setup.py install

edit:
	vim neat_gym/__init__.py

clean:
	rm -rf models/ visuals/

flake:
	flake8 neat_gym/__init__.py
	flake8 neat_gym/novelty/__init__.py
	flake8 neat-evolve.py
	flake8 neat-test.py
