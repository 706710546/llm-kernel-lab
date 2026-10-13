#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__device__ float to_float(scalar_t value) {
    return static_cast<float>(value);
}

template <typename scalar_t>
__device__ scalar_t from_float(float value) {
    return static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void rmsnorm_shared_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    std::int64_t n_columns,
    float eps) {
    const std::int64_t row = blockIdx.x;
    const scalar_t* row_input = input + row * n_columns;
    scalar_t* row_output = output + row * n_columns;
    __shared__ float partial[kThreads];

    // Each thread computes a local sum over its strided subset of the row.
    float local_square_sum = 0.0f;
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        const float value = to_float(row_input[column]);
        local_square_sum += value * value;
    }
    partial[threadIdx.x] = local_square_sum;
    __syncthreads();

    // A block-wide tree reduction combines 256 thread-local sums into one.
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float mean_square = partial[0] / static_cast<float>(n_columns);
    const float inverse_rms = rsqrtf(mean_square + eps);

    // Re-read input to avoid keeping an entire row live in registers.
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        row_output[column] = from_float<scalar_t>(
            to_float(row_input[column]) * inverse_rms * to_float(weight[column]));
    }
}

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename scalar_t>
__global__ void rmsnorm_shuffle_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    scalar_t* output,
    std::int64_t n_columns,
    float eps) {
    const std::int64_t row = blockIdx.x;
    const scalar_t* row_input = input + row * n_columns;
    scalar_t* row_output = output + row * n_columns;
    constexpr int kWarps = kThreads / 32;
    __shared__ float warp_sums[kWarps];

    float local_square_sum = 0.0f;
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        const float value = to_float(row_input[column]);
        local_square_sum += value * value;
    }

    // First reduce within each warp using register-to-register shuffle instructions.
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float warp_sum_value = warp_sum(local_square_sum);
    if (lane == 0) {
        warp_sums[warp] = warp_sum_value;
    }
    __syncthreads();

    // Warp 0 combines the eight warp-level sums; other lanes contribute the identity 0.
    float block_square_sum = 0.0f;
    if (warp == 0) {
        block_square_sum = lane < kWarps ? warp_sums[lane] : 0.0f;
        block_square_sum = warp_sum(block_square_sum);
        if (lane == 0) {
            warp_sums[0] = block_square_sum;
        }
    }
    __syncthreads();

    const float mean_square = warp_sums[0] / static_cast<float>(n_columns);
    const float inverse_rms = rsqrtf(mean_square + eps);
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        row_output[column] = from_float<scalar_t>(
            to_float(row_input[column]) * inverse_rms * to_float(weight[column]));
    }
}

template <typename scalar_t>
void launch_rmsnorm_kernel(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor& output,
    std::int64_t n_rows,
    std::int64_t n_columns,
    float eps,
    bool use_shuffle,
    cudaStream_t stream) {
    if (use_shuffle) {
        rmsnorm_shuffle_kernel<scalar_t><<<static_cast<unsigned int>(n_rows), kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            n_columns, eps);
    } else {
        rmsnorm_shared_kernel<scalar_t><<<static_cast<unsigned int>(n_rows), kThreads, 0, stream>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            n_columns, eps);
    }
}

}  // namespace

torch::Tensor run_rmsnorm_cuda(
    torch::Tensor input, torch::Tensor weight, double eps, bool use_shuffle) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16 ||
            input.scalar_type() == torch::kBFloat16,
        "input must be float32, float16, or bfloat16");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.size(-1) > 0, "normalized dimension must be non-empty");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
    TORCH_CHECK(weight.size(0) == input.size(-1), "weight length must match input's last dimension");
    TORCH_CHECK(weight.device() == input.device(), "input and weight must be on the same device");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "input and weight must have the same dtype");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0, "eps must be finite and non-negative");

    auto output = torch::empty_like(input);
    if (input.numel() == 0) {
        return output;
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    const std::int64_t n_columns = input.size(-1);
    const std::int64_t n_rows = input.numel() / n_columns;
    TORCH_CHECK(n_rows <= std::numeric_limits<int>::max(), "number of rows exceeds CUDA grid limit");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
    if (input.scalar_type() == torch::kFloat32) {
        launch_rmsnorm_kernel<float>(input, weight, output, n_rows, n_columns,
                                     static_cast<float>(eps), use_shuffle, stream);
    } else if (input.scalar_type() == torch::kFloat16) {
        launch_rmsnorm_kernel<c10::Half>(input, weight, output, n_rows, n_columns,
                                         static_cast<float>(eps), use_shuffle, stream);
    } else {
        launch_rmsnorm_kernel<c10::BFloat16>(input, weight, output, n_rows, n_columns,
                                             static_cast<float>(eps), use_shuffle, stream);
    }
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA RMSNorm launch failed: ", cudaGetErrorString(error));
    return output;
}

torch::Tensor rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
    return run_rmsnorm_cuda(input, weight, eps, false);
}

torch::Tensor rmsnorm_cuda_shuffle(torch::Tensor input, torch::Tensor weight, double eps) {
    return run_rmsnorm_cuda(input, weight, eps, true);
}
