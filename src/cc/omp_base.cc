// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include <omp.h>
#include "timer.h"
#define CC_VARIANT "omp_base"

void CCSolver(int m, int nnz, int *row_offsets, int *column_indices, CompT *comp) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP CC solver (%d threads) ...\n", num_threads);
	#pragma omp parallel for
	for (int n=0; n < m; n++) comp[n] = n;
	bool change = true;
	int iter = 0;

	Timer t;
	t.Start();
	while (change) {
		change = false;
		iter++;
		#pragma omp parallel for schedule(dynamic, 64)
		for (int src = 0; src < m; src ++) {
			int comp_src = comp[src];
			int row_begin = row_offsets[src];
			int row_end = row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = column_indices[offset];
				int comp_dst = comp[dst];      
				if ((comp_src < comp_dst) && (comp_dst == comp[comp_dst])) {
					change = true;
					comp[comp_dst] = comp_src;
				}
			}
		}
		#pragma omp parallel for
		for (int n = 0; n < m; n++) {
			while (comp[n] != comp[comp[n]]) {
				comp[n] = comp[comp[n]];
			}
		}
	}
	t.Stop();

	printf("\titerations = %d.\n", iter);
	printf("\truntime [%s] = %f ms.\n", CC_VARIANT, t.Millisecs());
	return;
}
