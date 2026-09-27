#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>


namespace {

constexpr int kThreads = 256;


__global__ void softmax_cuda_kernel(
    const float* input,
    float* output,
    std::int64_t n_columns) {
    const std::int64_t row = blockIdx.x;
    const float* row_input = input + row * n_columns;
    float* row_output = output + row * n_columns;
    __shared__ float partial[kThreads];

    // Each thread scans columns separated by the block width.
    float local_max = -CUDART_INF_F;
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        local_max = fmaxf(local_max, row_input[column]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce 256 -> 128 -> ... -> 1, synchronizing after each step.
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = partial[0];
    // Protect the read before reusing partial for the sum reduction.
    __syncthreads();

    float local_sum = 0.0f;
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        local_sum += expf(row_input[column] - row_max);
    }
    partial[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denominator = partial[0];

    // Re-read input instead of keeping a whole row live in registers.
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        row_output[column] = expf(row_input[column] - row_max) / denominator;
    }
}

template <bool IsMax>
__device__ float warp_reduce(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = IsMax ? fmaxf(value, other) : value + other;
    }
    return value;
}


template <bool IsMax>
__device__ float block_reduce(float value, float* partial) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    value = warp_reduce<IsMax>(value);
    if (lane == 0) {
        partial[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < kThreads / 32 ? partial[lane] : (IsMax ? -CUDART_INF_F : 0.0f);
        value = warp_reduce<IsMax>(value);
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    const float result = partial[0];
    // All threads must read the result before the buffer can be reused.
    __syncthreads();
    return result;
}


__global__ void softmax_cuda_shuffle_kernel(
    const float* input, float* output, std::int64_t n_columns) {
    const std::int64_t row = blockIdx.x;
    const float* row_input = input + row * n_columns;
    float* row_output = output + row * n_columns;
    __shared__ float partial[kThreads / 32];

    float local_max = -CUDART_INF_F;
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        local_max = fmaxf(local_max, row_input[column]);
    }
    const float row_max = block_reduce<true>(local_max, partial);
    float local_sum = 0.0f;
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        local_sum += expf(row_input[column] - row_max);
    }
    const float denominator = block_reduce<false>(local_sum, partial);
    for (std::int64_t column = threadIdx.x; column < n_columns; column += kThreads) {
        row_output[column] = expf(row_input[column] - row_max) / denominator;
    }
}

}  // namespace


torch::Tensor run_softmax_cuda(torch::Tensor input, bool shuffle) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    if (input.numel() == 0) {
        return output;
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    const std::int64_t n_rows = input.size(0);
    const std::int64_t n_columns = input.size(1);
    TORCH_CHECK(
        n_rows <= std::numeric_limits<unsigned int>::max(),
        "number of rows exceeds CUDA grid limit");
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
    if (shuffle) {
        softmax_cuda_shuffle_kernel<<<static_cast<unsigned int>(n_rows), kThreads, 0, stream>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), n_columns);
    } else {
        softmax_cuda_kernel<<<static_cast<unsigned int>(n_rows), kThreads, 0, stream>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), n_columns);
    }
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "softmax_cuda_kernel launch failed: ", cudaGetErrorString(error));
    return output;
}


torch::Tensor softmax_cuda(torch::Tensor input) {
    return run_softmax_cuda(input, false);
}


torch::Tensor softmax_cuda_shuffle(torch::Tensor input) {
    return run_softmax_cuda(input, true);
}
