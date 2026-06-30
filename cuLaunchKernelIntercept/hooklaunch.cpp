#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

// Real function pointers
static cudaError_t (*real_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t) = NULL;
static CUresult (*real_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**) = NULL;

typedef CUresult (*cuGetProcAddress_t)(const char*, void**, int, unsigned int, void*);
static cuGetProcAddress_t real_cuGetProcAddress = NULL;
static void* (*real_dlsym)(void*, const char*) = NULL;

extern "C" {

// 1. Your Custom Hook for cuLaunchKernel
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    if (real_cuLaunchKernel == NULL) {
        void* cu_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        real_cuLaunchKernel = (CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**))dlsym(cu_handle, "cuLaunchKernel");
    }
    fprintf(stderr, "[HOOK] Target Intercepted! cuLaunchKernel executed.\n");
    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

// 2. Your Custom Hook for cuGetProcAddress
CUresult my_cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, unsigned int flags, void* hipcuGetProcAddress) {
    fprintf(stderr, "[HOOK] Inside cuGetProcAddress looking for: %s\n", symbol);

    if (symbol && strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[HOOK] Hijacking cuLaunchKernel pointer assignment!\n");
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    }

    // Call the real cuGetProcAddress if it's anything else
    if (real_cuGetProcAddress == NULL) {
        void* cu_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        real_cuGetProcAddress = (cuGetProcAddress_t)dlvsym(RTLD_NEXT, "cuGetProcAddress", "RTLD_NEXT"); 
        if(!real_cuGetProcAddress) {
             real_cuGetProcAddress = (cuGetProcAddress_t)real_dlsym(cu_handle, "cuGetProcAddress");
        }
    }
    return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, hipcuGetProcAddress);
}

// 3. The dlsym Hook that starts the interception chain
void* dlsym(void* handle, const char* symbol) {
    // Bootstrap the real dlsym using dlvsym to avoid recursion
    if (real_dlsym == NULL) {
        real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym) {
            real_dlsym = (void* (*)(void*, const char*))dlopen("libdl.so.2", RTLD_NOW); 
        }
    }

    // CRUCIAL: Intercept libcudart trying to look up cuGetProcAddress
    if (symbol && (strcmp(symbol, "cuGetProcAddress_v2") == 0 || strcmp(symbol, "cuGetProcAddress") == 0)) {
        fprintf(stderr, "[HOOK] dlsym intercepted call for %s! Returning our hook.\n", symbol);
        
        // Save the real function pointer from the requested handle before we fake the return
        real_cuGetProcAddress = (cuGetProcAddress_t)real_dlsym(handle, symbol);
        
        return (void*)my_cuGetProcAddress;
    }

    return real_dlsym(handle, symbol);
}

}

/*

#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <syscall.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

cudaError_t (*kernel_func)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
CUresult (*kernel_func_cu)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra);

typedef CUresult (*cuGetProcAddress_t)(const char*, void**, int, unsigned int, void*);
static cuGetProcAddress_t real_cuGetProcAddress = NULL;

extern "C" {

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
	if (kernel_func == NULL) {
		void* cudart_handle = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
		*(void **)(&kernel_func) = dlsym (cudart_handle, "cudaLaunchKernel");
		assert (kernel_func != NULL);
	}

	fprintf(stderr, "caught call, cudaLaunchKernel\n");

    return (*kernel_func)(func, gridDim, blockDim, args, sharedMem, stream);
}

CUresult cuLaunchKernel (CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    if (kernel_func_cu == NULL) {
        void* cu_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        *(void **)(&kernel_func_cu) = dlsym (cu_handle, "cuLaunchKernel");
		assert (kernel_func_cu != NULL);
    }

    fprintf(stderr, "caught call, cuLaunchKernel\n");

    return (*kernel_func_cu)(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);

}

CUresult cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, unsigned int flags, void* hipcuGetProcAddress) {
    
    fprintf(stderr, "hooked - cuGetProcAddress or v2\n");

    // Fetch the real cuGetProcAddress from libcuda.so if we haven't yet
    if (real_cuGetProcAddress == NULL) {
        void* cu_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        real_cuGetProcAddress = (cuGetProcAddress_t)dlsym(cu_handle, "cuGetProcAddress");
        if (!real_cuGetProcAddress) {
            // Fallback to _v2 if necessary
            real_cuGetProcAddress = (cuGetProcAddress_t)dlsym(cu_handle, "cuGetProcAddress_v2");
        }
        assert(real_cuGetProcAddress != NULL);
    }

    // If the runtime is asking for cuLaunchKernel, give it OUR hook!
    if (symbol && strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[HOOK] Intercepted cuGetProcAddress request for: %s\n", symbol);
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    }

    // Otherwise, let the real CUDA driver handle the address resolution
    return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, hipcuGetProcAddress);
}

// Map cuGetProcAddress_v2 to the same hook just in case
CUresult cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, unsigned int flags, void* hipcuGetProcAddress) {
    return cuGetProcAddress(symbol, pfn, cudaVersion, flags, hipcuGetProcAddress);
}

}
*/