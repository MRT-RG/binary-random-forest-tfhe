# Makefile

CC = nvcc
CFLAGS = -std=c++11 -O3
INCLUDES = -I$(shell pwd)/cuFHE/cufhe
LIBS = -L$(shell pwd)/cuFHE/cufhe/bin -lcufhe_gpu

run: runme
	LD_LIBRARY_PATH=$(shell pwd)/cuFHE/cufhe/bin ./runme `cat testdata.txt | head -n 1`

runme: model.cu
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) -o runme main.cu model.cu

model.cu: model.pickle
	python3 code_generator.py --output model.cu

model.pickle:
	python3 train.py

clean:
	rm -rf model.pickle model.cu runme testdata.txt __pycache__

# vim: noexpandtab tabstop=4 shiftwidth=4
