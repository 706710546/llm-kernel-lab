#include <torch/extension.h>


torch::Tensor softmax_cuda(torch::Tensor input);
torch::Tensor softmax_cuda_shuffle(torch::Tensor input);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("softmax", &softmax_cuda, "Row-wise Softmax CUDA (FP32)");
    module.def("shuffle", &softmax_cuda_shuffle, "Warp Shuffle Softmax CUDA (FP32)");
}
