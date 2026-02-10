#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void block_reduce_sum(float *input, float *output, int n)
{
    __shared__ float sdata[512];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    if (i + blockDim.x < n)
    {
        sdata[tid] += input[i + blockDim.x];
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile float *smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

int main()
{
    const int n = 1 << 20; // 1M elements
    const int threads = 512;
    const int blocks = (n + threads * 2 - 1) / (threads * 2);

    float *h_input = (float *)malloc(n * sizeof(float));
    float *h_output = (float *)malloc(blocks * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    block_reduce_sum<<<blocks, threads>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; i++)
    {
        total += h_output[i];
    }

    printf("Expected: %f, Got: %f\n", (float)n, total);
    printf("Error: %f\n", fabsf(total - n));
    printf("%s\n", (fabsf(total - n) < 1.0f) ? "PASSED" : "FAILED");

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}