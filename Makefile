CXX = g++
CXXFLAGS = -fPIC -O2
LDFLAGS = -shared -ldl

TARGET = hooking.so
SRC = hooking.cpp

hooking.so:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

kernel_example.o: 
	nvcc -cudart=shared -std=c++11 -c -o kernel_example.o kernel_example.cu

threading: 
	g++ threading.cpp kernel_example.o -o threading -lcudart

all:
	make hooking.so
	make kernel_example.o
	make threading

clean:
	rm hooking.so
	rm kernel_example.o
	rm threading