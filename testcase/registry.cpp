#include "testcase.h"
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <limits>

namespace testcase {
static const Entry entries[] = {
    {"coverage", coverage, 4097}, {"vector_add", vector_add, 65536},
    {"matmul", matmul, 64}, {"compute", compute, 4096},
    {"chained", chained, 4097}, {"memset", memset, 4097},
    {"cublaslt_matmul", cublaslt_matmul, 64},
    {"cublaslt_chained", cublaslt_chained, 64}
};
const Entry* find(const std::string& name) {
    for (const auto& e : entries) if (name == e.name) return &e;
    return nullptr;
}
void list() {
    for (const auto& e : entries) std::printf("%s (default size=%zu)\n", e.name, e.default_size);
}
Selection parse(const std::string& spec) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    for (;;) {
        auto end = spec.find(':', start);
        parts.push_back(spec.substr(start, end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    const Entry* entry = find(parts[0]);
    if (!entry || parts.size() > 5) throw std::invalid_argument("Expected name[:size[:iterations[:work[:seed]]]]: " + spec);
    Config c;
    c.size = entry->default_size;
    unsigned long long values[4] = {c.size, 1, 256, 1};
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (parts[i].empty() || parts[i].find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("Expected unsigned decimal parameter: " + spec);
        values[i-1] = std::stoull(parts[i]);
    }
    // Bound sizes to keep the scheduler's signed 1D atom arithmetic valid.
    const auto max_size = (parts[0] == "matmul" || parts[0] == "cublaslt_matmul" || parts[0] == "cublaslt_chained") ? 1024ull : (1ull << 26);
    if (!values[0] || values[0] > max_size || !values[1] || values[1] > 1000000 ||
        !values[2] || values[2] > 1000000 || values[3] > UINT32_MAX)
        throw std::invalid_argument("Parameter outside supported range: " + spec);
    c.size = values[0]; c.iterations = values[1]; c.work = values[2]; c.seed = values[3];
    return {entry, c};
}
}
