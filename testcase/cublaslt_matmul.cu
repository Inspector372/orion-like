#include "cublaslt_common.cuh"

testcase::Result testcase::cublaslt_matmul(const Config& c) {
    return lt_detail::run(c, false);
}
