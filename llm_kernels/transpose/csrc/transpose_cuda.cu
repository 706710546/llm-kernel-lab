#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>


namespace {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;


__global__ void transpose_naive_kernel(
    const float* input,
    float* output,
    std::int64_t n_rows,
    std::int64_t n_columns) {
    const std::int64_t column =
        static_cast<std::int64_t>(blockIdx.x) * kTileDim + threadIdx.x;
    const std::int64_t first_row =
        static_cast<std::int64_t>(blockIdx.y) * kTileDim + threadIdx.y;

    #pragma unroll
    for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
        const std::int64_t row = first_row + offset;
        if (row < n_rows && column < n_columns) {
            output[column * n_rows + row] = input[row * n_columns + column];
        }
    }
}


__global__ void transpose_tiled_kernel(
    const float* input,
    float* output,
    std::int64_t n_rows,
    std::int64_t n_columns) {
    __shared__ float tile[kTileDim][kTileDim + 1];

    const int local_column = threadIdx.x;
    const int local_row = threadIdx.y;
    const std::int64_t input_column =
        static_cast<std::int64_t>(blockIdx.x) * kTileDim + local_column;
    const std::int64_t first_input_row =
        static_cast<std::int64_t>(blockIdx.y) * kTileDim + local_row;

    #pragma unroll
    for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
        const std::int64_t input_row = first_input_row + offset;
        if (input_row < n_rows && input_column < n_columns) {
            tile[local_row + offset][local_column] =
                input[input_row * n_columns + input_column];
        }
    }

    __syncthreads();

    const std::int64_t output_column =
        static_cast<std::int64_t>(blockIdx.y) * kTileDim + local_column;
    const std::int64_t first_output_row =
        static_cast<std::int64_t>(blockIdx.x) * kTileDim + local_row;

    #pragma unroll
    for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
        const std::int64_t output_row = first_output_row + offset;
        if (output_row < n_columns && output_column < n_rows) {
            output[output_row * n_rows + output_column] =
                tile[local_column][local_row + offset];
        }
    }
}


__global__ void transpose_tiled_unpadded_kernel(
    const float* input,
    float* output,
    std::int64_t n_rows,
    std::int64_t n_columns) {
    __shared__ float tile[kTileDim][kTileDim];

    const int local_column = threadIdx.x;
    const int local_row = threadIdx.y;
    const std::int64_t input_column =
        static_cast<std::int64_t>(blockIdx.x) * kTileDim + local_column;
    const std::int64_t first_input_row =
        static_cast<std::int64_t>(blockIdx.y) * kTileDim + local_row;

    #pragma unroll
    for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
        const std::int64_t input_row = first_input_row + offset;
        if (input_row < n_rows && input_column < n_columns) {
            tile[local_row + offset][local_column] =
                input[input_row * n_columns + input_column];
        }
    }

    __syncthreads();

    const std::int64_t output_column =
        static_cast<std::int64_t>(blockIdx.y) * kTileDim + local_column;
    const std::int64_t first_output_row =
        static_cast<std::int64_t>(blockIdx.x) * kTileDim + local_row;

    #pragma unroll
    for (int offset = 0; offset < kTileDim; offset += kBlockRows) {
        const std::int64_t output_row = first_output_row + offset;
        if (output_row < n_columns && output_column < n_rows) {
            output[output_row * n_rows + output_column] =
                tile[local_column][local_row + offset];
        }
    }
}


void check_input(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
}


torch::Tensor create_output(const torch::Tensor& input) {
    return torch::empty(
        {input.size(1), input.size(0)},
        input.options());
}


void check_launch(const char* kernel_name) {
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(
        error == cudaSuccess,
        kernel_name,
        " launch failed: ",
        cudaGetErrorString(error));
}

}  // namespace


torch::Tensor transpose_cuda_naive(torch::Tensor input) {
    check_input(input);
    auto output = create_output(input);
    if (input.numel() == 0) {
        return output;
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    const std::int64_t n_rows = input.size(0);
    const std::int64_t n_columns = input.size(1);
    const dim3 block(kTileDim, kBlockRows);
    const dim3 grid(
        static_cast<unsigned int>((n_columns + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((n_rows + kTileDim - 1) / kTileDim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

    transpose_naive_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n_rows,
        n_columns);
    check_launch("transpose_naive_kernel");
    return output;
}


torch::Tensor transpose_cuda_tiled(torch::Tensor input) {
    check_input(input);
    auto output = create_output(input);
    if (input.numel() == 0) {
        return output;
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    const std::int64_t n_rows = input.size(0);
    const std::int64_t n_columns = input.size(1);
    const dim3 block(kTileDim, kBlockRows);
    const dim3 grid(
        static_cast<unsigned int>((n_columns + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((n_rows + kTileDim - 1) / kTileDim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

    transpose_tiled_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n_rows,
        n_columns);
    check_launch("transpose_tiled_kernel");
    return output;
}


torch::Tensor transpose_cuda_tiled_unpadded(torch::Tensor input) {
    check_input(input);
    auto output = create_output(input);
    if (input.numel() == 0) {
        return output;
    }

    const c10::cuda::CUDAGuard device_guard(input.device());
    const std::int64_t n_rows = input.size(0);
    const std::int64_t n_columns = input.size(1);
    const dim3 block(kTileDim, kBlockRows);
    const dim3 grid(
        static_cast<unsigned int>((n_columns + kTileDim - 1) / kTileDim),
        static_cast<unsigned int>((n_rows + kTileDim - 1) / kTileDim));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());

    transpose_tiled_unpadded_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n_rows,
        n_columns);
    check_launch("transpose_tiled_unpadded_kernel");
    return output;
}
