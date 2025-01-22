#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000000
#define BLOCK_SIZE 256

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void add_cpu(float *a, float *b, float *c, int n) {
    for(int i=0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(h_a, d_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //warm-up
    for (int i =0; i < 20; i++) {
        add_cpu(h_a, h_b, h_c_cpu, N);
        add_gpu<<<blocks, BLOCK_SIZE>>>(d_a, d_b, h_c_gpu, N);
        cudaDeviceSynchronize();
    }

    int iter = 50;
    
    //CPU 
    double cpu_time = 0.0;
    for(int i = 0; i < iter; i++) {
        double start = get_time();
        add_cpu(h_a, h_b, h_c_cpu, N);
        double end = get_time();
        cpu_time += end - start;
    }
    double avg_cpu_time = cpu_time / iter;
    printf("CPU average time: %f\n", avg_cpu_time);

    //GPU
    double gpu_time = 0.0;
    for(int i = 0; i < iter; i++) {
        double start = get_time();
        add_gpu<<<blocks, BLOCK_SIZE>>>(d_a, d_b, h_c_gpu, N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_time += end - start;
    }
    double avg_gpu_time = gpu_time / iter;
    printf("GPU average time: %f\n", avg_gpu_time);

    printf("Speedup: %f\n", avg_cpu_time / avg_gpu_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
