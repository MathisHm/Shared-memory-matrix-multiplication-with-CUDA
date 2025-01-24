#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_mat(float *matrix, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = (float)rand() / RAND_MAX;
        }
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

    int innerRow = threadIdx.y / BLOCK_SIZE;
    int innerCol = threadIdx.x % BLOCK_SIZE;

    A += cRow * BLOCK_SIZE * K;
    B += cCol * BLOCK_SIZE;
    C += cRow * BLOCK_SIZE * K + cCol * BLOCK_SIZE;

    float temp = 0.0;
    for(int i=0; i < K; i+=BLOCK_SIZE) {
        ChunkA[innerRow * BLOCK_SIZE + innerCol] = A[innerRow * K + innerCol];
        ChunkB[innerRow * BLOCK_SIZE + innerCol] = B[innerRow * N + innerCol];
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        for(int j=0; j < BLOCK_SIZE; ++j) {
            temp += ChunkA[innerRow * BLOCK_SIZE + j] *
                    ChunkB[j * BLOCK_SIZE + innerCol];
        }
        __syncthreads();
    }

    C[innerRow * N + innerCol] = temp + C[innerRow * N + innerCol];
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

    init_mat(h_A, M, K);
    init_mat(h_B, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));

    double gpu_start = get_time();
    shared_memory_mat_mult<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    double gpu_end = get_time();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    double cpu_start = get_time();
    cpu_mat_mult(M, N, K, h_A, h_B, h_C_cpu);
    double cpu_end = get_time();

    printf("GPU time: %f seconds\n", gpu_end - gpu_start);
    printf("CPU time: %f seconds\n", cpu_end - cpu_start);
    printf("Speedup: %fx\n", (cpu_end - cpu_start) / (gpu_end - gpu_start));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
