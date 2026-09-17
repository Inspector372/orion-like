# Optional cuDNN C++ Frontend workloads (headers from NVIDIA/cudnn-frontend v1.9.0).
CXX=g++-12
ENABLE_CUDNN ?= 0
CUDNN_FRONTEND_DIR ?= ../cudnn-frontend
CUDNN_INCLUDE_DIR ?= /usr/include/x86_64-linux-gnu
CUDNN_LIB_DIR ?= /usr/lib/x86_64-linux-gnu
CUDNN_OBJECTS := testcase/cudnn_matmul.o testcase/cudnn_convolution.o testcase/cudnn_layernorm.o testcase/cudnn_attention.o
ifeq ($(ENABLE_CUDNN),1)
CUDNN_CPPFLAGS = -DORION_ENABLE_CUDNN=1 -I$(CUDNN_FRONTEND_DIR)/include -I$(CUDNN_INCLUDE_DIR)
CUDNN_LIBS = -L$(CUDNN_LIB_DIR) -lcudnn
else
CUDNN_CPPFLAGS = -DORION_ENABLE_CUDNN=0
endif

# Optional CUDA LibTorch distribution, alongside this project. No CMake needed.
ENABLE_LIBTORCH ?= 0
LIBTORCH_DIR ?= ../libtorch-cu126/libtorch
# GCC 10 reports __cplusplus=201709L even with -std=c++20.
# Use a newer host compiler for ATen; override with LIBTORCH_CXX=g++-13, etc.
LIBTORCH_CXX ?= g++-12
# Read the package ABI flag when provided; modern cxx11 packages default to 1.
LIBTORCH_CXX11_ABI ?= $(or $(shell sed -n 's/.*_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' "$(LIBTORCH_DIR)/share/cmake/Torch/TorchConfig.cmake" 2>/dev/null | head -n 1),1)
LIBTORCH_OBJECTS := testcase/libtorch_feedforward.o testcase/libtorch_convolution.o
ifeq ($(ENABLE_LIBTORCH),1)
ABI_FLAGS = -D_GLIBCXX_USE_CXX11_ABI=$(LIBTORCH_CXX11_ABI)
LIBTORCH_CPPFLAGS = -DORION_ENABLE_LIBTORCH=1 -I$(LIBTORCH_DIR)/include -I$(LIBTORCH_DIR)/include/torch/csrc/api/include
LIBTORCH_LIBS = -L$(LIBTORCH_DIR)/lib -Xlinker -rpath -Xlinker $(abspath $(LIBTORCH_DIR)/lib) -Xlinker --no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10 -Xlinker --as-needed
LIBTORCH_COMPILER = $(LIBTORCH_CXX)
LIBTORCH_STANDARD = c++20
# threading.cpp does not include ATen; nvcc does not need C++20 here.
LIBTORCH_LINK_FLAGS = -std=c++17
else
LIBTORCH_CPPFLAGS = -DORION_ENABLE_LIBTORCH=0
LIBTORCH_COMPILER = $(CXX)
LIBTORCH_STANDARD = c++17
endif

TEST_SOURCES := $(wildcard testcase/*.cu)
TEST_OBJECTS := $(TEST_SOURCES:.cu=.o) testcase/registry.o $(LIBTORCH_OBJECTS)

all: libsmctrl.o libsmctrl.a hooking.so wrapper.o threading

libsmctrl.o:
	gcc libsmctrl.c -c -o libsmctrl.o -fPIC -lcuda -L/usr/local/cuda-12.8/lib64/stubs

libsmctrl.a:
	ar rcs libsmctrl.a libsmctrl.o

hooking.so: hooking.cpp hooking.h
	$(CXX) $(ABI_FLAGS) -fPIC hooking.cpp -o hooking.so -shared -ldl -I/usr/local/cuda-12.8/include

# -G option is important, this ignores some compiler optimization,
# which leads to failure of kernel-inside-kernel launch.
wrapper.o: wrapper.cu wrapper.h
	nvcc $(ABI_FLAGS) -G -cudart=shared -std=c++11 -arch=sm_70 -c -o wrapper.o wrapper.cu

# Testcases should work for multiple architectures, but that's for later.
testcase/%.o: testcase/%.cu testcase/testcase.h testcase/common.cuh
	nvcc $(ABI_FLAGS) -arch=sm_70 -c $< -o $@

testcase/registry.o: testcase/registry.cpp testcase/testcase.h
	$(CXX) $(ABI_FLAGS) -std=c++11 -c $< -o $@

testcase/cublaslt_matmul.o testcase/cublaslt_chained.o: testcase/cublaslt_common.cuh

threading: threading.cpp $(TEST_OBJECTS) wrapper.o libsmctrl.o
	nvcc $(ABI_FLAGS) $(LIBTORCH_LINK_FLAGS) -G -g -Xcompiler -pthread threading.cpp $(TEST_OBJECTS) wrapper.o libsmctrl.o -o threading -ldl -L/usr/local/cuda-12.8/lib64 -lcudart -lcuda -lcublasLt $(CUDNN_LIBS) $(LIBTORCH_LIBS) -L/usr/local/cuda-12.8/lib64/stubs

clean:
	rm -f libsmctrl.o libsmctrl.a hooking.so wrapper.o threading.o threading testcase/*.o .testcase-build-config .testcase-build-config.tmp

# Rebuild these four objects when switching optional dependency settings.
.PHONY: cudnn-force
cudnn-force:
$(CUDNN_OBJECTS): testcase/%.o: testcase/%.cu testcase/cudnn_common.cuh testcase/cudnn_reference.h testcase/common.cuh testcase/testcase.h cudnn-force
ifeq ($(ENABLE_CUDNN),1)
	@test -f "$(CUDNN_FRONTEND_DIR)/include/cudnn_frontend.h" || (echo "Set CUDNN_FRONTEND_DIR to the cudnn-frontend checkout"; exit 1)
endif
	nvcc $(ABI_FLAGS) -std=c++17 -arch=sm_70 $(CUDNN_CPPFLAGS) -c $< -o $@

# LibTorch headers contain host C++ code; compile with g++, not nvcc.
$(LIBTORCH_OBJECTS): testcase/%.o: testcase/%.cpp testcase/libtorch_common.h testcase/testcase.h
ifeq ($(ENABLE_LIBTORCH),1)
	@test -f "$(LIBTORCH_DIR)/include/ATen/ATen.h" -a -f "$(LIBTORCH_DIR)/lib/libtorch_cuda.so" || (echo "Set LIBTORCH_DIR to a CUDA-enabled LibTorch distribution"; exit 1)
endif
	$(LIBTORCH_COMPILER) $(CXXFLAGS) -std=$(LIBTORCH_STANDARD) -pthread $(ABI_FLAGS) $(LIBTORCH_CPPFLAGS) -c $< -o $@

.PHONY: check-libtorch-compiler
check-libtorch-compiler:
ifeq ($(ENABLE_LIBTORCH),1)
	@command -v $(LIBTORCH_CXX) >/dev/null 2>&1 || (echo "LibTorch compiler $(LIBTORCH_CXX) not found; install GCC 12+ or set LIBTORCH_CXX to a compatible compiler"; exit 1)
	@printf '%s\n' '#if __cplusplus < 202002L' '#error ATen requires C++20: use LIBTORCH_CXX=g++-12 or newer, not g++-10' '#endif' | $(LIBTORCH_CXX) $(CXXFLAGS) -std=c++20 -x c++ -fsyntax-only -
endif
$(LIBTORCH_OBJECTS): | check-libtorch-compiler

# All C++ objects exchanging testcase::Result/std::string must use the same ABI.
# Update the stamp only when settings change, so LibTorch is not rebuilt on
# every make. Switching back to ENABLE_LIBTORCH=0 also invalidates old objects.
.PHONY: all clean testcase-config-force
testcase-config-force:
.testcase-build-config: testcase-config-force
	@printf '%s\n' '$(ENABLE_LIBTORCH)' '$(LIBTORCH_CPPFLAGS)' '$(ABI_FLAGS)' '$(LIBTORCH_LIBS)' '$(CXX)' '$(CXXFLAGS)' '$(LIBTORCH_COMPILER)' '$(LIBTORCH_STANDARD)' '$(LIBTORCH_LINK_FLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || cp $@.tmp $@
	@rm -f $@.tmp
$(TEST_OBJECTS) wrapper.o hooking.so threading: .testcase-build-config
