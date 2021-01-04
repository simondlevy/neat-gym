install:
	sudo python3 setup.py install

edit:
	vim neat_gym/__init__.py

nedit:
	vim neat_gym/novelty/__init__.py

clean:
	rm -rf models/ visuals/

commit:
	git commit -a

flake:
	flake8 neat_gym/__init__.py
	flake8 neat_gym/novelty/__init__.py
	flake8 neat-evolve.py
	flake8 neat-test.py
