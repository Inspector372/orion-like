libsmctrl.o:
	gcc libsmctrl.c -c -o libsmctrl.o -fPIC

libsmctrl.a:
	ar rcs libsmctrl.a libsmctrl.o

hooklaunch.so:
	g++ -fPIC hooklaunch.cpp -o hooklaunch.so -shared -ldl

test_argumod.cu: 
	nvcc -g -G test_argumod.cu -o test_argumod libsmctrl.a -ldl -lcudart -lcuda -L/usr/local/cuda-12.8/lib64/stubs

all:
	make libsmctrl.o
	make libsmctrl.a
	make hooklaunch.so
	make test_argumod.cu

clean:
	rm libsmctrl.o
	rm libsmctrl.a
	rm hooklaunch.so
	rm test_argumod