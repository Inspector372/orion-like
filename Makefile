TEST_SOURCES := $(wildcard testcase/*.cu)
TEST_OBJECTS := $(TEST_SOURCES:.cu=.o) testcase/registry.o

all: libsmctrl.o libsmctrl.a hooking.so wrapper.o threading

libsmctrl.o:
	gcc libsmctrl.c -c -o libsmctrl.o -fPIC -lcuda -L/usr/local/cuda-12.8/lib64/stubs

libsmctrl.a:
	ar rcs libsmctrl.a libsmctrl.o

hooking.so: 
	g++ -fPIC hooking.cpp -o hooking.so -shared -ldl -I/usr/local/cuda-12.8/include

# -G option is important, this ignores some compiler optimization,
# which leads to failure of kernel-inside-kernel launch.
wrapper.o: 
	nvcc -G -cudart=shared -std=c++11 -arch=sm_70 -c -o wrapper.o wrapper.cu

# Testcases should work for multiple architectures, but that's for later.
testcase/%.o: testcase/%.cu testcase/testcase.h testcase/common.cuh
	nvcc -arch=sm_70 -c $< -o $@

testcase/registry.o: testcase/registry.cpp testcase/testcase.h
	g++ -std=c++11 -c $< -o $@

testcase/cublaslt_matmul.o testcase/cublaslt_chained.o: testcase/cublaslt_common.cuh

threading: threading.cpp $(TEST_OBJECTS) wrapper.o libsmctrl.o
	nvcc -G -g -Xcompiler -pthread threading.cpp $(TEST_OBJECTS) wrapper.o libsmctrl.o -o threading -ldl -lcudart -lcuda -lcublasLt -L/usr/local/cuda-12.8/lib64/stubs

clean:
	rm -f libsmctrl.o libsmctrl.a hooking.so wrapper.o threading.o threading testcase/*.o