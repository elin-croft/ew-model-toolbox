OS := $(shell uname -s)
ifeq ($(OS), Darwin)
	PLATFORM := macos
else ifeq ($(OS), Linux)
	PLATFORM := linux
else ifeq ($(OS), Windows)
	PLATFORM := windows
else
	PLATFORM := unknown
endif

ifeq ($(PLATFORM), macos)
	TAR_NO_XATTRS := --no-xattrs
else
	TAR_NO_XATTRS :=
endif

info:
	@echo "OS: $(OS)"
	@echo "Platform: $(PLATFORM)"

tar:
	tar $(TAR_NO_XATTRS) -czf toolbox.tar.gz \
		--exclude='toolbox.tar.gz' \
		--exclude='*.DS_Store*' \
		--exclude='.git*' \
		--exclude='.idea*' \
		--exclude='tmp*' \
		--exclude='*.log' \
		--exclude='*.onnx' \
		--exclude='*.pth' \
		.

thrift:
	thrift -r --gen py thrift_file/toolbox_thrift/toolbox.thrift
	rm -rf gen_py
	mv gen-py gen_py

clean:
	find . -name "*.pyc"|xargs rm
	find . -name "*pycache*"|xargs rm -r

pack: clean thrift tar