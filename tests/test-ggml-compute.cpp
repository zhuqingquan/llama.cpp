#include <stdio.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

int main(int /*argc*/, char* /*argv*/[]) {
    const int rows = 4;
    const int cols = 4;
    const float scalar = 0.5f;  // multiplier in [0, 1.0]

    // ---- 1) init ggml context (data lives in the pool, no_alloc=false) ----
    struct ggml_init_params params = {
        /*.mem_size   =*/ 128 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ctx = ggml_init(params);

    // ---- 2) create a 4x4 float matrix and fill it ----
    struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    {
        float* data = (float*)a->data;
        for (int i = 0; i < rows * cols; i++) {
            data[i] = (float)(i + 1);  // 1.0, 2.0, ..., 16.0
        }
    }

    // mark a as input so the allocator does not overwrite it
    ggml_set_input(a);

    // ---- 3) build compute graph: result = a * scalar ----
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    struct ggml_tensor* result = ggml_scale(ctx, a, scalar);
    ggml_build_forward_expand(gf, result);

    // ---- 4) execute on CPU backend ----
    ggml_backend_t backend = ggml_backend_init_by_name("CPU", nullptr);
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_graph_compute(backend, gf);

    // ---- 5) print results ----
    printf("Input matrix (%d x %d):\n", rows, cols);
    const float* in = (const float*)a->data;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", in[i * cols + j]);
        }
        printf("\n");
    }

    printf("\nScaled by %.1f:\n", scalar);
    const float* out = (const float*)result->data;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", out[i * cols + j]);
        }
        printf("\n");
    }

    // ---- 6) verify correctness ----
    for (int i = 0; i < rows * cols; i++) {
        float expected = in[i] * scalar;
        assert(fabs(out[i] - expected) < 1e-5f);
    }
    printf("\nResult verified: all %d elements correct.\n", rows * cols);

    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
