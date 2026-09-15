# Optional cuDNN C++ Frontend workloads (headers from NVIDIA/cudnn-frontend v1.9.0).
ENABLE_CUDNN ?= 0
CUDNN_FRONTEND_DIR ?= /opt/cudnn-frontend
CUDNN_INCLUDE_DIR ?= /usr/local/cuda-12.8/include
CUDNN_LIB_DIR ?= /usr/local/cuda-12.8/lib64
CUDNN_OBJECTS := testcase/cudnn_matmul.o testcase/cudnn_convolution.o testcase/cudnn_layernorm.o testcase/cudnn_attention.o
ifeq ($(ENABLE_CUDNN),1)
CUDNN_CPPFLAGS = -DORION_ENABLE_CUDNN=1 -I$(CUDNN_FRONTEND_DIR)/include -I$(CUDNN_INCLUDE_DIR)
CUDNN_LIBS = -L$(CUDNN_LIB_DIR) -lcudnn
else
CUDNN_CPPFLAGS = -DORION_ENABLE_CUDNN=0
endif

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
	nvcc -G -g -Xcompiler -pthread threading.cpp $(TEST_OBJECTS) wrapper.o libsmctrl.o -o threading -ldl -lcudart -lcuda -lcublasLt $(CUDNN_LIBS) -L/usr/local/cuda-12.8/lib64/stubs

clean:
	rm -f libsmctrl.o libsmctrl.a hooking.so wrapper.o threading.o threading testcase/*.o

# Rebuild these four objects when switching optional dependency settings.
.PHONY: cudnn-force
cudnn-force:
$(CUDNN_OBJECTS): testcase/%.o: testcase/%.cu testcase/cudnn_common.cuh testcase/cudnn_reference.h testcase/common.cuh testcase/testcase.h cudnn-force
ifeq ($(ENABLE_CUDNN),1)
	@test -f "$(CUDNN_FRONTEND_DIR)/include/cudnn_frontend.h" || (echo "Set CUDNN_FRONTEND_DIR to the cudnn-frontend checkout"; exit 1)
endif
	nvcc -std=c++17 -arch=sm_70 $(CUDNN_CPPFLAGS) -c $< -o $@
