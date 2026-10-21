#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

static ggml_backend_t fault_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, params);
    GGML_ASSERT(backend != nullptr);
    if (const char * after = std::getenv("PERF_ABORT_AFTER")) {
        static int remaining;
        remaining = std::atoi(after);
        auto set_abort = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_set_abort_callback");
        GGML_ASSERT(set_abort != nullptr);
        set_abort(backend, [](void * data) { return (*static_cast<int *>(data))-- <= 0; }, &remaining);
    }
    return backend;
}

static ggml_status fault_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    fprintf(stderr, "FAULT_GRAPH_STATUS=%s\n", ggml_status_to_string(status));
    return status;
}

static bool fault_supports(ggml_backend_t backend, const ggml_tensor * op) {
    return std::getenv("PERF_UNSUPPORTED") == nullptr && ggml_backend_supports_op(backend, op);
}

#define ggml_backend_dev_init fault_init
#define ggml_backend_graph_compute fault_compute
#define ggml_backend_supports_op fault_supports
#include "test-source.cpp"
