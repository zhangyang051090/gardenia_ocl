// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "sgd.h"
#include "graph_io.h"
#include <thrust/sort.h>
#include <random>
void Initialize(int len, LatentT *lv) {
	std::default_random_engine rng;
	std::uniform_real_distribution<float> dist(0, 0.1);
	for (int i = 0; i < len; ++i) {
		for (int j = 0; j < K; ++j) {
			lv[i*K+j] = dist(rng);
		}
	}
	/*
	for (int i = 0; i < m; i++) {
		unsigned r = i;
		for (int j = 0; j < K; j++)
			init_user_lv[i*K+j] = ((LatentT)rand_r(&r)/(LatentT)RAND_MAX);
	}
	for (int i = 0; i < n; i++) {
		unsigned r = i + m;
		for (int j = 0; j < K; j++)
			init_item_lv[i*K+j] = ((LatentT)rand_r(&r)/(LatentT)RAND_MAX);
	}
	*/
}

int main(int argc, char *argv[]) {
	ScoreT lambda = 0.05; // regularization_factor
	ScoreT step = 0.003; // learning_rate
	int max_iters = 1;
	float epsilon = 0.1;
	printf("Stochastic Gradient Descent by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [lambda(0.05)] [step(0.003)] [max_iter(1)]\n", argv[0]);
		exit(1);
	}
	if (argc > 2) lambda = atof(argv[2]);
	if (argc > 3) step = atof(argv[3]);
	if (argc > 4) max_iters = atof(argv[4]);
	if (argc > 5) epsilon = atof(argv[5]);
	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, false, false, false, false, false);
	printf("num_users=%d, num_items=%d\n", m, n);
	printf("regularization_factor=%f, learning_rate=%f\n", lambda, step);
	printf("max_iters=%d, epsilon=%f\n", max_iters, epsilon);

	LatentT *h_user_lv = (LatentT *)malloc(m * K * sizeof(LatentT));
	LatentT *h_item_lv = (LatentT *)malloc(n * K * sizeof(LatentT));
	LatentT *lv_u = (LatentT *)malloc(m * K * sizeof(LatentT));
	LatentT *lv_i = (LatentT *)malloc(n * K * sizeof(LatentT));
	ScoreT *h_rating = (ScoreT *) malloc(nnz * sizeof(ScoreT));

	//srand(0);
	Initialize(m, lv_u);
	Initialize(n, lv_i);
	for (int i = 0; i < m * K; i++) h_user_lv[i] = lv_u[i];
	for (int i = 0; i < n * K; i++) h_item_lv[i] = lv_i[i];
	for (int i = 0; i < nnz; i++) h_rating[i] = (ScoreT)h_weight[i];
	printf("Shuffling users...\n");
	int *ordering = (int *)malloc(m * sizeof(int));
	thrust::sequence(ordering, ordering+m);
	std::random_shuffle(ordering, ordering+m);

	SGDSolver(m, n, nnz, h_row_offsets, h_column_indices, h_rating, h_user_lv, h_item_lv, lambda, step, ordering, max_iters, epsilon);
	SGDVerifier(m, n, nnz, h_row_offsets, h_column_indices, h_rating, lv_u, lv_i, lambda, step, ordering, max_iters, epsilon);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	free(h_weight);
	free(h_user_lv);
	free(h_item_lv);
	free(lv_u);
	free(lv_i);
	free(h_rating);
	return 0;
}
