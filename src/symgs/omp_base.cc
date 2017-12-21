// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "symgs.h"
#include <omp.h>
#include "timer.h"
#define SYMGS_VARIANT "omp_base"

void gauss_seidel(int *Ap, int *Aj, int *indices, ValueType *Ax, ValueType *x, ValueType *b, int row_start, int row_stop, int row_step) {
	//printf("Solving, num_rows=%d\n", row_stop-row_start);
	#pragma omp parallel for
	for (int i = row_start; i < row_stop; i += row_step) {
		int inew = indices[i];
		int row_begin = Ap[inew];
		int row_end = Ap[inew+1];
		ValueType rsum = 0;
		ValueType diag = 0;
		#pragma ivdep
		for (int jj = row_begin; jj < row_end; jj++) {
			const int j = Aj[jj];  //column index
			if (inew == j) diag = Ax[jj];
			else rsum += x[j] * Ax[jj];
		}
		if (diag != 0) x[inew] = (b[inew] - rsum) / diag;
	}
}

void SymGSSolver(int num_rows, int nnz, int *Ap, int *Aj, int *indices, ValueType *Ax, ValueType *x, ValueType *b, std::vector<int> color_offsets) {
	int num_threads = 1;
#pragma omp parallel
	{
		num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP SymGS solver (%d threads) ...\n", num_threads);
	Timer t;
	t.Start();
	for(size_t i = 0; i < color_offsets.size()-1; i++)
		gauss_seidel(Ap, Aj, indices, Ax, x, b, color_offsets[i], color_offsets[i+1], 1);
	for(size_t i = color_offsets.size()-1; i > 0; i--)
		gauss_seidel(Ap, Aj, indices, Ax, x, b, color_offsets[i-1], color_offsets[i], 1);
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", SYMGS_VARIANT, t.Millisecs());
	return;
}
