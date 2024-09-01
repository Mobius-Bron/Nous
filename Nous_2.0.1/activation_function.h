#include <stdio.h>
#include <cuda_runtime.h>

namespace actvation_fun{
    __global__ void ReLU(float* input_tensor, float* output_tensor, const int N){
        const int gid = threadIdx.x + blockIdx.x * blockDim.x;
        if(gid >= N){return;}
        output_tensor[gid] = input_tensor[gid] > 0 ? input_tensor[gid] : 0;
    }

    __global__ void Sigmoid(float* input_tensor, float* output_tensor, const int N) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        if (gid < N) {
            output_tensor[gid] = 1.0f / (1.0f + expf(-input_tensor[gid]));
        }
    }

    __global__ void Tanh(float* input_tensor, float* output_tensor, const int N) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        if (gid < N) {
            output_tensor[gid] = tanhf(input_tensor[gid]);
        }
    }

    __global__ void LeakyReLU(float* input_tensor, float* output_tensor, const int N) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        if (gid < N) {
            output_tensor[gid] = input_tensor[gid] > 0 ? input_tensor[gid] : 0.01f * input_tensor[gid];
        }
    }

    __global__ void PReLU(float alpha, float* input_tensor, float* output_tensor, const int N) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        if (gid < N) {
            output_tensor[gid] = input_tensor[gid] > 0 ? input_tensor[gid] : alpha * input_tensor[gid];
        }
    }

    __global__ void ELU(float alpha, float* input_tensor, float* output_tensor, const int N) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        if (gid < N) {
            if (input_tensor[gid] > 0) {
                output_tensor[gid] = input_tensor[gid];
            } else {
                output_tensor[gid] = alpha * (expf(input_tensor[gid]) - 1.0f);
            }
        }
    }

    __global__ void Softplus(float* input_tensor, float* output_tensor, const int N) {
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        if (gid < N) {
            output_tensor[gid] = logf(1.0f + expf(input_tensor[gid]));
        }
    }
}

