#include "testcase.h"
#include <cstdio>
#include <exception>
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--list") { testcase::list(); return 0; }
    if (argc != 2) { std::fprintf(stderr, "Usage: %s name[:size[:iterations[:work[:seed]]]] | --list\n", argv[0]); return 2; }
    try {
        auto selected = testcase::parse(argv[1]);
        auto result = selected.entry->run(selected.config);
        std::printf("%s: %s: %s\n", selected.entry->name, result.passed ? "PASS" : "FAIL", result.message.c_str());
        return result.passed ? 0 : 1;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
