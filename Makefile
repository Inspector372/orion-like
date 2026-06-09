
libsmctrl.o:
	gcc libsmctrl.c -c -o libsmctrl.o -fPIC -I/usr/local/cuda-12.8/targets/x86_64-linux/include

libsmctrl.a:
	ar rcs libsmctrl.a libsmctrl.o

hooking.so:
	g++ -fPIC hooking.cpp -o hooking.so -shared -ldl

kernel_example.o: 
	nvcc -cudart=shared -std=c++11 -c -o kernel_example.o kernel_example.cu

wrapper.o: 
	nvcc -cudart=shared -std=c++11 -c -o wrapper.o wrapper.cu

threading: 
	g++ threading.cpp kernel_example.o wrapper.o -o threading libsmctrl.a -ldl -pthread -lcudart -lcuda -L/usr/local/cuda-12.8/lib64/stubs

all:
	make libsmctrl.o
	make libsmctrl.a
	make hooking.so
	make kernel_example.o
	make wrapper.o
	make threading

clean:
	rm libsmctrl.o 
	rm libsmctrl.a
	rm hooking.so
	rm kernel_example.o
	rm wrapper.o
	rm threading