#include <cuda_runtime.h>
#include <stdio.h>

template <typename T>
__global__ void kernel_e(T *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n - 1)
    {
        T temp = data[idx];
        data[idx] = data[idx + 1];
        data[idx + 1] = temp;
    }
}

int main()
{
    const int n = 1000;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    int *h_data = new int[n];
    int *h_expected = new int[n];

    // Initialize
    for (int i = 0; i < n; i++)
    {
        h_data[i] = i;
        h_expected[i] = i;
    }

    // Expected: swap adjacent pairs
    for (int i = 0; i < n - 1; i++)
    {
        int temp = h_expected[i];
        h_expected[i] = h_expected[i + 1];
        h_expected[i + 1] = temp;
    }

    int *d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    kernel_e<<<blocks, threads>>>(d_data, n);

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
                break;
        }
    }

    printf("%s\n", correct ? "PASSED" : "FAILED");

    delete[] h_data;
    delete[] h_expected;
    cudaFree(d_data);

    return 0;
}