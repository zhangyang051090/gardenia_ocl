// Copyright 2016, National University of Defense Technology
// Contact: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
/*
GARDINIA Benchmark Suite
Kernel: Sparse Matrix-Vector Multiplication (SpMV)
Author: Xuhao Chen 

Will return vector y

This SpMV implements optimizations from Bell et al.[1] on GPU.

[1] Nathan Bell and Michael Garland, Implementing Sparse Matrix-Vector 
    Multiplication on Throughput-Oriented Processors, SC'09

[2] Samuel Williams et. al, Optimization of Sparse Matrix-Vector 
	Multiplication on Emerging Multicore Platforms, SC'07

[3] Xing Liu et. al, Efficient Sparse Matrix-Vector Multiplication
	on x86-Based Many-Core Processors, ICS’13
    
spmv_omp : one thread per row (vertex) using OpenMP
spmv_base: one thread per row (vertex) using CUDA
spmv_warp: one warp per row (vertex) using CUDA
spmv_vector: one vector per row (vertex) using CUDA
*/

#define ValueType float
void SpmvSolver(int m, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y);
void SpmvVerifier(int m, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y, ValueType *test_y);
