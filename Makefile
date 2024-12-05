clean:
	find . -name "*.pyc"|xargs rm
	find . -name "*pycache*"|xargs rm -r