// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "pr.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("PageRank by Xuhao Chen\n");
	bool is_directed = true;
	bool symmetrize = false;
	if (argc < 2) {
		printf("Usage: %s <graph> [is_directed(0/1)]\n", argv[0]);
		exit(1);
	} else if (argc> 2) {
		is_directed = atoi(argv[2]);
		if(is_directed) printf("This is a directed graph\n");
		else printf("This is an undirected graph\n");
	}
	if(!is_directed) symmetrize = true;

	int m, n, nnz;//, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	int *in_row_offsets, *out_row_offsets, *in_column_indices, *out_column_indices, *in_degree, *out_degree;
	read_graph(argc, argv, m, n, nnz, out_row_offsets, out_column_indices, out_degree, h_weight, symmetrize, false, false);
	read_graph(argc, argv, m, n, nnz, in_row_offsets, in_column_indices, in_degree, h_weight, symmetrize, true, false);

	ScoreT *h_scores = (ScoreT *) malloc(m * sizeof(ScoreT));
	const ScoreT init_score = 1.0f / m;
	for (int i = 0; i < m; i++) h_scores[i] = init_score;
	PRSolver(m, nnz, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, out_degree, h_scores);
	PRVerifier(m, in_row_offsets, in_column_indices, out_row_offsets, out_column_indices, out_degree, h_scores, EPSILON);

	free(in_row_offsets);
	free(in_column_indices);
	free(in_degree);
	free(out_row_offsets);
	free(out_column_indices);
	free(out_degree);
	free(h_scores);
	free(h_weight);
	return 0;
}
