.PHONY: build
build: clean
	python setup.py build_ext --inplace

clean: 
	rm -f -r build/
	rm -f sgcc/lhrr/lhrr.cpp
