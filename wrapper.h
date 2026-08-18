#include "cuco/static_map.cuh"

typedef struct alignas(16) AtomMetaData{
    uint64_t kernel; 
    uint32_t lidx;
    uint32_t hidx; 

    __host__ __device__ bool operator==(const MyValue& other) const {
        return kernel == other.kernel && 
               lidx == other.lidx && 
               hidx == other.hidx;
    }
} AtomMetaData;

namespace cuco {
template <>
struct is_bitwise_comparable<AtomMetaData> : std::true_type {};
}

typedef void (*func_ptr_t)();

__global__ void wrapper(const __grid_constant__ uint32_t argu);

extern CUfunction wrapper_handle;
extern cuco::static_map<uint64_t, AtomMetaData>* atomMetaDataTable;

void initial_wrapper_run();
void initial_nothing_run();