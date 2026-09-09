#include "cublaslt_common.cuh"

testcase::Result testcase::cublaslt_chained(const Config& c) {
    return lt_detail::run(c, true);
}
