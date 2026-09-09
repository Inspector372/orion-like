#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace testcase {
struct Config {
    std::size_t size = 4097; // Elements; square matrix dimension for matmul.
    int iterations = 1;     // Number of kernel/chain repetitions.
    int work = 256;         // Recurrence steps for compute; fill/kernel pairs per iteration for memset.
    std::uint32_t seed = 1;
};
struct Result { bool passed; std::string message; };
using Function = Result (*)(const Config&);
struct Entry { const char* name; Function run; std::size_t default_size; };
struct Selection { const Entry* entry; Config config; };
Result coverage(const Config&);
Result vector_add(const Config&);
Result matmul(const Config&);
Result compute(const Config&);
Result chained(const Config&);
Result memset(const Config&);
const Entry* find(const std::string& name);
void list();
// name[:size[:iterations[:work[:seed]]]]; throws on invalid input.
Selection parse(const std::string& spec);
}
