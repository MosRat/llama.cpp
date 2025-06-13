//
// Created by whl on 2025/6/13.
//

#include <chrono>
#include <cstdio>

#include "gex.h"

int main() {
    constexpr auto model_path = R"(D:\whl\Desktop\Gext_Pt1-596M-Q4_K_M.gguf)";
    constexpr auto mm_path    = R"(D:\whl\Desktop\mmproj-gext_pt1.gguf)";
    constexpr auto img_path   = R"(D:\whl\Desktop\test1.png)";

    auto * ctx = gex_init_default(model_path, mm_path);
    for (int i = 0; i < 5; i++) {
        auto         start = std::chrono::high_resolution_clock::now();
        const auto * res   = gex_inference_path(ctx, img_path);
        if (res == nullptr) {
            std::printf("[Failed] %s", get_last_error().msg);
        }
        auto end      = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::printf("result is %s\nuse %lld ms\n", res, duration.count() / 1000);
        fflush(stdout);
    }

    gex_free(ctx);
    return 0;
}
