#include "testcase.h"
#include <cassert>
#include <exception>
namespace testcase {
Result coverage(const Config&) { return {true,""}; }
Result vector_add(const Config&) { return {true,""}; }
Result matmul(const Config&) { return {true,""}; }
Result compute(const Config&) { return {true,""}; }
Result chained(const Config&) { return {true,""}; }
}
int main() {
    auto m = testcase::parse("matmul"); assert(m.config.size == 64);
    auto c = testcase::parse("compute:1025:3:4096:0");
    assert(c.config.size == 1025 && c.config.iterations == 3 && c.config.work == 4096 && c.config.seed == 0);
    assert(testcase::find("chained") && !testcase::find("missing"));
    const char* invalid[] = {"", "missing", "coverage:", "coverage:0", "coverage:-1", "coverage:1:0", "matmul:1025", "compute:1:1:0", "coverage:1:1:1:4294967296", "coverage:1:1:1:1:1", "coverage:184467440737095516160", "coverage:2x", "coverage:67108865"};
    for (auto s : invalid) { bool rejected=false; try { testcase::parse(s); } catch (const std::exception&) { rejected=true; } assert(rejected); }
}
