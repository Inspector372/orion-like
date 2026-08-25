
typedef struct AtomMetaData {
    uint64_t key;
    uint64_t kernel; 
    uint32_t lidx;
    uint32_t hidx; 

} AtomMetaData;

typedef void (*func_ptr_t)();

__global__ void wrapper(const __grid_constant__ uint32_t argu);

extern CUfunction wrapper_handle;
extern void table_insert(uint64_t, AtomMetaData);
extern void setup_metadata();
extern cudaStream_t metadata_pass_stream;

void initial_wrapper_run();
void initial_nothing_run();