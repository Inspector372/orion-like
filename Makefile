.DEFAULT_GOAL := all
CUDA_HOME ?= /usr/local/cuda-12.8
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_70
# Keep -G: the project's injected kernel entry currently relies on debug codegen.
CUDAFLAGS ?= -G -g -std=c++11 -arch=$(ARCH) -cudart=shared
CPPFLAGS += -I$(CUDA_HOME)/include
CUDA_LIBS = -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcudart -lcuda
TEST_SOURCES := $(wildcard testcase/*.cu)
TEST_OBJECTS := $(TEST_SOURCES:.cu=.o) testcase/registry.o

.PHONY: all clean test-host
all: hooking.so threading testcase_runner

libsmctrl.o: libsmctrl.c libsmctrl.h
	$(CC) $(CPPFLAGS) -fPIC -c $< -o $@
libsmctrl.a: libsmctrl.o
	$(AR) rcs $@ $^
hooking.so: hooking.cpp hooking.h
	$(CXX) $(CPPFLAGS) -std=c++11 -fPIC -shared $< -o $@ -ldl -pthread
wrapper.o: wrapper.cu wrapper.h
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
testcase/%.o: testcase/%.cu testcase/testcase.h testcase/common.cuh
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
testcase/%.o: testcase/%.cpp testcase/testcase.h
	$(CXX) -std=c++11 -Wall -Wextra -c $< -o $@
threading: threading.cpp hooking.h wrapper.h libsmctrl.h testcase/testcase.h $(TEST_OBJECTS) wrapper.o libsmctrl.o
	$(NVCC) $(CUDAFLAGS) -x cu -c threading.cpp -o threading.o -Xcompiler -pthread
	$(NVCC) $(CUDAFLAGS) threading.o $(TEST_OBJECTS) wrapper.o libsmctrl.o -o $@ -Xcompiler -pthread -ldl $(CUDA_LIBS)
testcase_runner: testcase/standalone.o $(TEST_OBJECTS)
	$(NVCC) $(CUDAFLAGS) $^ -o $@
clean:
	rm -f libsmctrl.o libsmctrl.a hooking.so wrapper.o threading.o threading testcase_runner testcase_registry_test testcase/*.o

# Host-only parser/registry checks; workload stubs do not exercise CUDA.
test-host:
	$(CXX) -std=c++11 -Wall -Wextra -Werror testcase/registry.cpp testcase/registry_test.cpp -o testcase_registry_test
	./testcase_registry_test
