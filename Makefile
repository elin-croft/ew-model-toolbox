thrift:
	thrift -r --gen py thrift_file/toolbox/toolbox.thrift
	rm -rf gen_py
	mv gen-py gen_py

clean:
	find . -name "*.pyc"|xargs rm
	find . -name "*pycache*"|xargs rm -r