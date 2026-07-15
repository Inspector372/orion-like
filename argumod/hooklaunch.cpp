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
static cuGetProcAddress_t real_cuGetProcAddress_v2 = NULL;
static void* (*real_dlsym)(void*, const char*) = NULL;
static void* cu_handle = NULL;

extern "C" {

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, 
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, 
                        unsigned int sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) {
    if (real_cuLaunchKernel == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuLaunchKernel = (CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**))real_dlsym(cu_handle, "cuLaunchKernel");
        if(real_cuLaunchKernel == NULL) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernel == NULL\n");
        }
        if(real_cuLaunchKernel == cuLaunchKernel) {
            fprintf(stderr, "FATAL ERROR: real_cuLaunchKernel == cuLaunchKernel\n");
        }
    }
    fprintf(stderr, "[HOOK] Target Intercepted! cuLaunchKernel executed.\n");
    return real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult my_cuGetProcAddress(const char* symbol, void** pfn, int cudaVersion, unsigned int flags, void* symbolStatus) {
    fprintf(stderr, "[HOOK v1] Inside cuGetProcAddress looking for: %s\n", symbol);

    if (symbol && strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[HOOK v1] Hijacking cuLaunchKernel pointer assignment!\n");
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    }

    // Call the real cuGetProcAddress if it's anything else
    if (real_cuGetProcAddress == NULL) {
        if(cu_handle == NULL) cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
        real_cuGetProcAddress = (cuGetProcAddress_t)dlvsym(RTLD_NEXT, "cuGetProcAddress", "RTLD_NEXT"); 
        if(!real_cuGetProcAddress) {
             real_cuGetProcAddress = (cuGetProcAddress_t)real_dlsym(cu_handle, "cuGetProcAddress");
        }
    }
    return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

CUresult my_cuGetProcAddress_v2(const char* symbol, void** pfn, int cudaVersion, unsigned int flags, void* symbolStatus) {
    fprintf(stderr, "[HOOK v2] Inside cuGetProcAddress_v2 looking for: %s\n", symbol);

    if (symbol && strcmp(symbol, "cuLaunchKernel") == 0) {
        fprintf(stderr, "[HOOK v2] Hijacking cuLaunchKernel pointer assignment!\n");
        *pfn = (void*)cuLaunchKernel;
        return CUDA_SUCCESS;
    }

    if (symbol && strcmp(symbol, "cuGetProcAddress") == 0) {
        fprintf(stderr, "[HOOK v2] Hijacking cuGetProcAddress pointer assignment!\n");
        *pfn = (void*)my_cuGetProcAddress;
        return CUDA_SUCCESS;
    }

    // Call the real cuGetProcAddress if it's anything else
    if (real_cuGetProcAddress_v2 == NULL) {
        if(cu_handle == NULL) {
            cu_handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if(cu_handle == NULL) {
                fprintf(stderr, "[HOOK v2] WARNING: cu_handle == NULL\n");
            }
        }
        real_cuGetProcAddress_v2 = (cuGetProcAddress_t)dlvsym(RTLD_NEXT, "cuGetProcAddress_v2", "RTLD_NEXT"); 
        if(!real_cuGetProcAddress_v2) {
             real_cuGetProcAddress_v2 = (cuGetProcAddress_t)real_dlsym(cu_handle, "cuGetProcAddress_v2");
        }
        if(real_cuGetProcAddress_v2 == NULL) {
            fprintf(stderr, "[HOOK v2] WARNING: real_cuGetProcAddress_v2 == NULL\n");
        }
    }
    return real_cuGetProcAddress(symbol, pfn, cudaVersion, flags, symbolStatus);
}

void* dlsym(void* handle, const char* symbol) {
    fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s.\n", symbol);
    // Bootstrap the real dlsym using dlvsym to avoid recursion
    if (real_dlsym == NULL) {
        real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (real_dlsym == NULL) {
            real_dlsym = (void* (*)(void*, const char*))dlopen("libdl.so.2", RTLD_NOW); 
        }
    }

    // CRUCIAL: Intercept libcudart trying to look up cuGetProcAddress
    if (symbol && (strcmp(symbol, "cuGetProcAddress") == 0)) {
        fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s! Returning our hook.\n", symbol);
        
        // Save the real function pointer from the requested handle before we fake the return
        real_cuGetProcAddress = (cuGetProcAddress_t)real_dlsym(handle, symbol);
        
        return (void*)my_cuGetProcAddress;
    }

    if (symbol && (strcmp(symbol, "cuGetProcAddress_v2") == 0)) {
        fprintf(stderr, "[HOOK dlsym] dlsym intercepted call for %s! Returning our hook.\n", symbol);
        
        // Save the real function pointer from the requested handle before we fake the return
        real_cuGetProcAddress = (cuGetProcAddress_t)real_dlsym(handle, symbol);
        
        return (void*)my_cuGetProcAddress_v2;
    }

    return real_dlsym(handle, symbol);
}

}
