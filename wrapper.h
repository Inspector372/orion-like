typedef struct{
    unsigned char data[128];  
} box128;

typedef void (*func_ptr_t)();

extern "C" void run_wrapper();