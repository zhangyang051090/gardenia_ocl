// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <omp.h>
#include "timer.h"
#include "omp_target_config.h"
#define SPMV_VARIANT "omp_target"

void SpmvSolver(int num_rows, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y) {
	int num_threads = 1;
	#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	//printf("Launching OpenMP SpMV solver (%d threads) ...\n", num_threads);
	warm_up();
	Timer t;
	t.Start();
	double t1, t2;
	#pragma omp target data device(0) map(tofrom:y[0:num_rows]) map(to:num_rows,Ap[0:(num_rows+1)],x[0:num_rows]) map(to:Aj[0:nnz],Ax[0:nnz])
	//#pragma omp target device(0) map(tofrom:y[0:num_rows]) map(to:num_rows,Ap[0:(num_rows+1)],x[0:num_rows]) map(to:Aj[0:nnz],Ax[0:nnz])
	{
	t1 = omp_get_wtime();
	#pragma omp target device(0)
	#pragma omp parallel for
	for (int i = 0; i < num_rows; i++){
		int row_begin = Ap[i];
		int row_end   = Ap[i+1];
		ValueType sum = y[i];
		#pragma ivdep
		//#pragma omp simd
		for (int jj = row_begin; jj < row_end; jj++) {
			const int j = Aj[jj];  //column index
			sum += x[j] * Ax[jj];
		}
		y[i] = sum; 
	}
	t2 = omp_get_wtime();
	}
	t.Stop();
	//printf("\ttotal runtime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, 1000*(t2-t1));
	return;
}
