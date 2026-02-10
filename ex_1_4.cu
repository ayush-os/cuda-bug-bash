#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_d(float *input, float *output, int n)
{
    __shared__ float shared[32][32];

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    shared[tid / 32][tid % 32] = input[bid * 1024 + tid];
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < 32; i++)
    {
        sum += shared[tid % 32][i];
    }

    output[bid * 1024 + tid] = sum;
}

int main()
{
    const int blocks = 128;
    const int threads = 1024;
    const int n = blocks * threads;

    float *h_input = new float[n];
    float *h_output = new float[n];

    for (int i = 0; i < n; i++)
    {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    kernel_d<<<blocks, threads>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
    {
        kernel_d<<<blocks, threads>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time per kernel: %f us\n", milliseconds * 1000 / 1000);

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++)
    {
        float expected = 32.0f; // Each thread sums 32 values of 1.0
        if (fabs(h_output[i] - expected) > 1e-3)
        {
            printf("Mismatch at %d: expected %f, got %f\n", i, expected, h_output[i]);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "PASSED" : "FAILED");

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}