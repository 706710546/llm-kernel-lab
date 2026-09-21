#include <torch/extension.h>


torch::Tensor transpose_cuda_naive(torch::Tensor input);
torch::Tensor transpose_cuda_tiled_unpadded(torch::Tensor input);
torch::Tensor transpose_cuda_tiled(torch::Tensor input);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("naive", &transpose_cuda_naive, "Naive Matrix Transpose CUDA (FP32)");
    module.def(
        "tiled_unpadded",
        &transpose_cuda_tiled_unpadded,
        "Tiled Matrix Transpose CUDA without padding (FP32)");
    module.def("tiled", &transpose_cuda_tiled, "Tiled Matrix Transpose CUDA (FP32)");
}
