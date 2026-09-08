
hooklaunch.so:
	g++ -fPIC hooklaunch.cpp -o hooklaunch.so -shared -ldl

kernel: 
	nvcc kernel.cu -o kernel -ldl -lcudart -lcuda -L/usr/local/cuda-12.8/lib64/stubs

all:
	make hooklaunch.so
	make kernel

clean:
	rm hooklaunch.so
	rm kernel