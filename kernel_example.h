#include <pthread.h>

typedef struct addKernel_arg {
    int N;
    int* h_A;
    int* h_B;
    int* h_out;
    pthread_mutex_t* smutex;

} addKernel_arg;

extern "C" void* addKernel_wrap(void* arg);
