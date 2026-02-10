#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void kernel_b(float *A, float *B, float *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float a = A[idx];
        float b = B[idx];
        C[idx] = sqrtf(a * a + b * b);
    }
}

int main()
{
    const int n = 1 << 24; // 16M elements
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];

    for (int i = 0; i < n; i++)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(i + 1);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    kernel_b<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        kernel_b<<<blocks, threads>>>(d_A, d_B, d_C, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float bandwidth = (3.0f * n * sizeof(float) * 100) / (milliseconds / 1000.0f) / 1e9;

    printf("Time: %f ms\n", milliseconds / 100);
    printf("Bandwidth: %f GB/s\n", bandwidth);

    // Get theoretical peak
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float theoretical_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;

    printf("Theoretical BW: %f GB/s\n", theoretical_bw);
    printf("Efficiency: %.2f%%\n", (bandwidth / theoretical_bw) * 100);

    // Verify correctness
    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < 100; i++)
    {
        float expected = sqrtf(h_A[i] * h_A[i] + h_B[i] * h_B[i]);
        if (fabs(h_C[i] - expected) > 1e-3)
        {
            printf("Mismatch at %d: expected %f, got %f\n", i, expected, h_C[i]);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "PASSED" : "FAILED");

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}