#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel_c(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        if (data[idx] % 2 == 0)
        {
            data[idx] = data[idx] / 2;
        }
        else
        {
            data[idx] = 3 * data[idx] + 1;
        }
    }
}

int main()
{
    const int n = 1000;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    int *h_data = new int[n];
    int *h_expected = new int[n];

    // Initialize with sequence 1..1000
    for (int i = 0; i < n; i++)
    {
        h_data[i] = i + 1;
        h_expected[i] = i + 1;
    }

    // Compute expected on CPU
    for (int i = 0; i < n; i++)
    {
        if (h_expected[i] % 2 == 0)
        {
            h_expected[i] = h_expected[i] / 2;
        }
        else
        {
            h_expected[i] = 3 * h_expected[i] + 1;
        }
    }

    int *d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    kernel_c<<<blocks, threads>>>(d_data, n);

    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify
    bool correct = true;
    for (int i = 0; i < n; i++)
    {
        if (h_data[i] != h_expected[i])
        {
            printf("Mismatch at %d: expected %d, got %d\n", i, h_expected[i], h_data[i]);
            correct = false;
            if (i > 10)
                break; // Only show first few errors
        }
    }

    printf("%s\n", correct ? "PASSED" : "FAILED");

    delete[] h_data;
    delete[] h_expected;
    cudaFree(d_data);

    return 0;
}