#define RELEASE_QUEUE_LENGTH 16384

typedef struct AtomMetaData {
    uint64_t key;
    uint64_t kernel; 
    uint32_t lidx;
    uint32_t hidx; 

} AtomMetaData;

typedef struct ReleaseQueueEntry {
    uint64_t key;
    cudaEvent_t event;
    bool reaped;

} ReleaseQueueEntry;

typedef void (*func_ptr_t)();

__global__ void wrapper(const __grid_constant__ uint64_t argu);

extern CUfunction wrapper_handle;
extern void table_insert(uint64_t, AtomMetaData);
extern void table_delete(uint64_t);
extern void release_queue_insert(uint64_t, cudaEvent_t);
extern void release_queue_delete(uint64_t);
void release_queue_mark_as_reaped(uint64_t);
extern void setup_metadata();
extern cudaStream_t metadata_pass_stream;
extern cudaError_t (*actual_cudaDeviceSynchronize)(void);

extern ReleaseQueueEntry release_queue[];

void initial_wrapper_run();
