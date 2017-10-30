CC=g++
CFLAGS=-I include
TARGETS=src/chai.cpp src/fcl.cpp src/layer.cpp src/loadmnist.cpp src/relu.cpp src/sigmoid.cpp src/softmax.cpp

make:
	$(CC) $(CFLAGS) -c $(TARGETS) -O3

clean:
	rm *.o

test:
	$(CC) $(CFLAGS) examples/MNIST_TEST.cpp *.o -o test -O3
