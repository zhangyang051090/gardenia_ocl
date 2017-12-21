// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "bc.h"
#include "graph_io.h"

int main(int argc, char *argv[]) {
	printf("Betweenness Centrality by Xuhao Chen\n");
	int source = 0;
	bool is_directed = true;
	bool symmetrize = false;
	if (argc < 2) {
		printf("Usage: %s <graph> [source_id] [is_directed(0/1)]\n", argv[0]);
		exit(1);
	} else if (argc> 2) {
		source = atoi(argv[2]);
		printf("Source vertex: %d\n", source);
		if(argc>3) {
			is_directed = atoi(argv[3]);
			if(is_directed) printf("This is a directed graph\n");
			else printf("This is an undirected graph\n");
		}
	}
	if(!is_directed) symmetrize = true;

	int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	WeightT *h_weight = NULL;
	read_graph(argc, argv, m, n, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, symmetrize);

	ScoreT *h_scores = (ScoreT *)malloc(m * sizeof(ScoreT));
	for (int i = 0; i < m; i++) h_scores[i] = 0;
	BCSolver(m, nnz, source, h_row_offsets, h_column_indices, h_scores);
	BCVerifier(m, source, h_row_offsets, h_column_indices, 1, h_scores);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	free(h_weight);
	free(h_scores);//zy
	return 0;
}
