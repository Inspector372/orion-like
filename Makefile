CXX = g++
CXXFLAGS = -fPIC -O2
LDFLAGS = -shared -ldl

TARGET = hooking.so
SRC = hooking.cpp

libsmctrl.o:
	gcc libsmctrl.c -c -o libsmctrl.o -fPIC

libsmctrl.a:
	ar rcs libsmctrl.a libsmctrl.o

hooking.so:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

kernel_example.o: 
	nvcc -cudart=shared -std=c++11 -c -o kernel_example.o kernel_example.cu

wrapper.o: 
	nvcc -cudart=shared -std=c++11 -c -o wrapper.o wrapper.cu

threading: 
	g++ threading.cpp kernel_example.o -o threading libsmctrl.a -ldl -pthread -lcudart -L/usr/local/cuda-12.8/lib64

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