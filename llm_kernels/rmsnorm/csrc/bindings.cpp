#include <torch/extension.h>

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, double eps);
torch::Tensor rmsnorm_cuda_shuffle(torch::Tensor input, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rmsnorm", &rmsnorm_cuda, "Row-wise RMSNorm CUDA (shared-memory reduction)");
    module.def("rmsnorm_shuffle", &rmsnorm_cuda_shuffle, "Row-wise RMSNorm CUDA (Warp Shuffle reduction)");
}
