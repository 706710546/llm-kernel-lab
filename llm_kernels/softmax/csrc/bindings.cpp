#include <torch/extension.h>


torch::Tensor softmax_cuda(torch::Tensor input);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("softmax", &softmax_cuda, "Row-wise Softmax CUDA (FP32)");
}
