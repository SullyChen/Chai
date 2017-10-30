CC=g++
TARGETS=../src/chai.cpp ../src/conv2d.cpp ../src/fcl.cpp ../src/layer.cpp ../src/loadmnist.cpp ../src/relu.cpp ../src/sigmoid.cpp ../src/softmax.cpp

make:
	mkdir bin; cd bin; $(CC) -I ../include -c $(TARGETS) -O3; cd ..

clean:
	rm bin/*.o

test:
	$(CC) -I include examples/MNIST_TEST.cpp bin/*.o -o test -O3
