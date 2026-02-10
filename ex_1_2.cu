#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

struct Vector3
{
    float x, y, z;
};

__global__ void vector_magnitude(Vector3 *vectors, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        Vector3 v = vectors[idx];
        result[idx] = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }
}

int main()
{
    const int n = 1 << 24;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    Vector3 *h_vectors = (Vector3 *)malloc(n * sizeof(Vector3));
    float *h_result = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++)
    {
        h_vectors[i].x = 1.0f;
        h_vectors[i].y = 2.0f;
        h_vectors[i].z = 3.0f;
    }

    Vector3 *d_vectors;
    float *d_result;
    cudaMalloc(&d_vectors, n * sizeof(Vector3));
    cudaMalloc(&d_result, n * sizeof(float));

    cudaMemcpy(d_vectors, h_vectors, n * sizeof(Vector3), cudaMemcpyHostToDevice);

    vector_magnitude<<<blocks, threads>>>(d_vectors, d_result, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        vector_magnitude<<<blocks, threads>>>(d_vectors, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float bytes = (sizeof(Vector3) + sizeof(float)) * n * 100;
    float bandwidth = bytes / (milliseconds / 1000.0f) / 1e9;

    printf("Time: %f ms\n", milliseconds / 100);
    printf("Bandwidth: %f GB/s\n", bandwidth);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float theoretical_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("Theoretical: %f GB/s\n", theoretical_bw);
    printf("Efficiency: %.1f%%\n", (bandwidth / theoretical_bw) * 100);

    cudaMemcpy(h_result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = sqrtf(1.0f + 4.0f + 9.0f);
    printf("Correctness: %s\n", (fabs(h_result[0] - expected) < 0.01) ? "PASSED" : "FAILED");

    free(h_vectors);
    free(h_result);
    cudaFree(d_vectors);
    cudaFree(d_result);

    return 0;
}