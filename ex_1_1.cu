#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void reduce_sum(float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int n = 10000;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    
    float* h_input = (float*)malloc(n * sizeof(float));
    float* h_output = (float*)malloc(blocks * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    
    float* d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    reduce_sum<<<blocks, threads>>>(d_input, d_output, n);
    
    cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total += h_output[i];
    }
    
    printf("Expected: %f, Got: %f\n", (float)n, total);
    printf("%s\n", (fabs(total - n) < 0.1) ? "PASSED" : "FAILED");
    
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}