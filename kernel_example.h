#include <pthread.h>

typedef struct addKernel_arg {
    int N;
    int* h_A;
    int* h_B;
    int* h_out;
    pthread_mutex_t* smutex;

} addKernel_arg;

extern "C" void* addKernel_wrap(void* arg);

extern "C" void* dotKernel_wrap(void* arg);

extern "C" void* matMulKernel_wrap(void* arg);

extern "C" void* histKernel_wrap(void* arg);

extern "C" void* softmaxKernel_wrap(void* arg);

extern "C" void* test_cublas(void* arg);

extern "C" void* chainedKernels_wrap(void* arg);