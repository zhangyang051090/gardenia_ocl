// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
#include "immintrin.h"
#include "timer.h"
#include <omp.h>

void run_serial(int n, ValueType *a, ValueType *b, ValueType *c) {
	Timer t;
	t.Start();
	for(int i = 0; i < n; i ++) {
		c[i] = a[i] + b[i];
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", "serial", t.Millisecs());
}

void run_simd(int n, ValueType *a, ValueType *b, ValueType *c) {
	Timer t;
	t.Start();
	for(int i = 0; i < n; i += 4) {
		__m128 ai = _mm_load_ps(a + i);
		__m128 bi = _mm_load_ps(b + i);
		__m128 ci = _mm_add_ps(ai, bi);
		_mm_store_ps(c + i, ci);
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", "simd", t.Millisecs());
}

void run_omp(int n, ValueType *a, ValueType *b, ValueType *c) {
	int num_threads = 1;
	#pragma omp parallel
	{
	num_threads = omp_get_num_threads();
	}
	printf("Launching OpenMP solver (%d threads) ...\n", num_threads);
	
	Timer t;
	t.Start();
	#pragma omp parallel for
	for(int i = 0; i < n; i ++) {
		c[i] = a[i] + b[i];
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", "omp", t.Millisecs());
}

void run_omp_simd(int n, ValueType *a, ValueType *b, ValueType *c) {
	Timer t;
	t.Start();
	#pragma omp parallel for simd
	for(int i = 0; i < n; i ++) {
		c[i] = a[i] + b[i];
	}
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", "omp", t.Millisecs());
}

int main(int argc, char *argv[]) {
	int num = 1024 * 1024;
	if(argc == 2) num = atoi(argv[1]);
	printf("Vector Addition num = %d\n", num);
	ValueType *h_a = (ValueType *)malloc(num * sizeof(ValueType));
	ValueType *h_b = (ValueType *)malloc(num * sizeof(ValueType));
	ValueType *h_c = (ValueType *)malloc(num * sizeof(ValueType));
	for(int i = 0; i < num; i ++) { h_a[i] = 1; h_b[i] = 1; h_c[i] = 0; }
	printf("Lauching serial...\n");
	run_serial(num, h_a, h_b, h_c);
	for(int i = 0; i < num; i ++) { h_a[i] = 1; h_b[i] = 1; h_c[i] = 0; }
	printf("Lauching simd...\n");
	run_simd(num, h_a, h_b, h_c);
	for(int i = 0; i < num; i ++) { h_a[i] = 1; h_b[i] = 1; h_c[i] = 0; }
	printf("Lauching omp...\n");
	run_omp(num, h_a, h_b, h_c);
	for(int i = 0; i < num; i ++) { h_a[i] = 1; h_b[i] = 1; h_c[i] = 0; }
	printf("Lauching omp_simd...\n");
	run_omp_simd(num, h_a, h_b, h_c);
	//run_gpu(num, h_a, h_b, h_c);

	//for(int i = 0; i < 16; i ++) { printf("c[%d]=%f\n", i, h_c[i]); }
	free(h_a);
	free(h_b);
	free(h_c);
	return 0;
}
