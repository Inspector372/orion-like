typedef struct{
    unsigned char data[256];  
} box256;

typedef void (*func_ptr_t)();

extern "C" void run_wrapper();