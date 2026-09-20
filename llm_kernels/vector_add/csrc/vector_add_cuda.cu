#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>


namespace {

__global__ void vector_add_kernel(
    const float* x,
    const float* y,
    float* output,
    std::int64_t n_elements) {
    const std::int64_t index =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n_elements) {
        output[index] = x[index] + y[index];
    }
}

}  // namespace


torch::Tensor vector_add_cuda(
    torch::Tensor x,
    torch::Tensor y,
    std::int64_t threads_per_block) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.device() == y.device(), "x and y must be on the same device");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.scalar_type() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(
        threads_per_block >= 32 && threads_per_block <= 1024 &&
            (threads_per_block & (threads_per_block - 1)) == 0,
        "threads_per_block must be a power of two in [32, 1024]");

    auto output = torch::empty_like(x);
    const std::int64_t n_elements = x.numel();
    if (n_elements == 0) {
        return output;
    }

    const c10::cuda::CUDAGuard device_guard(x.device());
    const int blocks = static_cast<int>(
        (n_elements + threads_per_block - 1) / threads_per_block);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());

    vector_add_kernel<<<blocks, static_cast<int>(threads_per_block), 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        n_elements);

    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        "vector_add_kernel launch failed: ",
        cudaGetErrorString(error));
    return output;
}
