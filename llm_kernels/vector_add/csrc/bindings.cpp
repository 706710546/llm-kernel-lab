#include <torch/extension.h>

#include <cstdint>


torch::Tensor vector_add_cuda(
    torch::Tensor x,
    torch::Tensor y,
    std::int64_t threads_per_block);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("vector_add", &vector_add_cuda, "Vector Add CUDA (FP32)");
}
