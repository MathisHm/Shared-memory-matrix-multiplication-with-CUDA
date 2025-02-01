#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_mat(float *matrix, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

void cpu_mat_mult(int M, int N, int K, const float *A, const float *B, float *C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

__global__ void shared_memory_mat_mult(int M, int N, int K, 
                                       const float *A, const float *B, float *C) {
    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    __shared__ float ChunkA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float ChunkB[BLOCK_SIZE * BLOCK_SIZE];

    int innerRow = threadIdx.y;
    int innerCol = threadIdx.x;

    A += cRow * BLOCK_SIZE * K;
    B += cCol * BLOCK_SIZE;
    C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;

    float temp = 0.0;
    for(int i=0; i < K; i+=BLOCK_SIZE) {
        ChunkA[innerRow * BLOCK_SIZE + innerCol] = A[innerRow * K + innerCol];
        ChunkB[innerRow * BLOCK_SIZE + innerCol] = B[innerRow * N + innerCol];
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        for(int j=0; j < BLOCK_SIZE; ++j) {
            temp += ChunkA[innerRow * BLOCK_SIZE + j] * ChunkB[j * BLOCK_SIZE + innerCol];
        }
        __syncthreads();
    }

    C[innerRow * N + innerCol] = temp;
}

int main() {
    int M = 2048;
    int N = 2048;
    int K = 2048;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_cpu = (float *)malloc(sizeC);
    float *h_C_cublas = (float *)malloc(sizeC);

    init_mat(h_A, M, K);
    init_mat(h_B, K, N);

    float *d_A, *d_B, *d_C, *d_C_cublas;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMalloc(&d_C_cublas, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));

    // GPU
    double gpu_start = get_time();
    shared_memory_mat_mult<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    double gpu_end = get_time();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // CPU
    double cpu_start = get_time();
    cpu_mat_mult(M, N, K, h_A, h_B, h_C_cpu);
    double cpu_end = get_time();

    // cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    
    double cublas_start = get_time();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                N, M, K, 
                &alpha, 
                d_B, N, 
                d_A, K, 
                &beta, 
                d_C_cublas, N);
    cudaDeviceSynchronize();
    double cublas_end = get_time();

    cudaMemcpy(h_C_cublas, d_C_cublas, sizeC, cudaMemcpyDeviceToHost);
    
    printf("CPU time: %f seconds\n", cpu_end - cpu_start);
    printf("GPU time: %f seconds\n", gpu_end - gpu_start);
    printf("cuBLAS time: %f seconds\n", cublas_end - cublas_start);
    printf("Speedup (CUDA Kernel vs. CPU): %fx\n", (cpu_end - cpu_start) / (gpu_end - gpu_start));
    printf("Speedup (cuBLAS vs. CUDA Kernel): %fx\n", (gpu_end - gpu_start) / (cublas_end - cublas_start));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    free(h_C_cublas);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_cublas);
    cublasDestroy(handle);

    return 0;
}
