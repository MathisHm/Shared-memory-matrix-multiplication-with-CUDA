# Shared Memory Matrix Multiplication with CUDA

This project implements matrix multiplication using CUDA, leveraging shared memory for improved performance compared to traditional parallel implementations. The code also compares execution times between the CUDA and CPU versions.

## Compilation
Compile with `nvcc`:
```bash
nvcc matrix_multiplication.cu -o matrix_multiplication
