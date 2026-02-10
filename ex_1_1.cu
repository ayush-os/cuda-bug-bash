#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_a(float *input, float *output, int n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        sdata[tid] = input[idx];
    }
    // Missing initialization for out-of-bounds threads

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main()
{
    const int n = 1000;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    float *h_input = new float[n];
    float *h_output = new float[blocks];

    // Initialize input
    for (int i = 0; i < n; i++)
    {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    kernel_a<<<blocks, threads>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result
    float total = 0.0f;
    for (int i = 0; i < blocks; i++)
    {
        total += h_output[i];
    }

    printf("Expected: %f, Got: %f\n", (float)n, total);

    if (fabs(total - n) < 1e-3)
    {
        printf("PASSED\n");
    }
    else
    {
        printf("FAILED\n");
    }

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}