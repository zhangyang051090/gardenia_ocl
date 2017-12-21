// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
#include "pr.h"
#include "timer.h"
#include <vector>
#include <stdlib.h>

void PRVerifier(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *degree, ScoreT *scores_to_test, double target_error) {
	printf("Verifying...\n");
	const ScoreT base_score = (1.0f - kDamp) / m;
	const ScoreT init_score = 1.0f / m;
	ScoreT *scores = (ScoreT *) malloc(m * sizeof(ScoreT));
	for (int i = 0; i < m; i ++) scores[i] = init_score;
	vector<ScoreT> outgoing_contrib(m);
	int iter;
	Timer t;
	t.Start();
	for (iter = 0; iter < MAX_ITER; iter ++) {
		double error = 0;
		for (int n = 0; n < m; n ++)
			outgoing_contrib[n] = scores[n] / degree[n];
		for (int src = 0; src < m; src ++) {
			ScoreT incoming_total = 0;
			int row_begin = in_row_offsets[src];
			int row_end = in_row_offsets[src + 1];
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = in_column_indices[offset];
				incoming_total += outgoing_contrib[dst];
			}
			ScoreT old_score = scores[src];
			scores[src] = base_score + kDamp * incoming_total;
			error += fabs(scores[src] - old_score);
		}   
		printf(" %2d    %lf\n", iter+1, error);
		if (error < EPSILON) break;
	}
	t.Stop();
	printf("\titerations = %d.\n", iter+1);
	printf("\truntime [serial] = %f ms.\n", t.Millisecs());

	//for(int i = 0; i < m; i++){
	//	if(scores_to_test[i] !=scores[i]){
	//		printf("failed: scores not the same in %d\n", i);
	//		break;
	//	}
	//}
	
	float *incomming_sums = (float *)malloc(m * sizeof(float));
	for(int i = 0; i < m; i ++) incomming_sums[i] = 0;
	double error = 0;
	for (int src = 0; src < m; src ++) {
		float outgoing_contrib = scores_to_test[src] / degree[src];
		int row_begin = out_row_offsets[src];
		int row_end = out_row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = out_column_indices[offset];
			incomming_sums[dst] += outgoing_contrib;
		}
	}
	for (int i = 0; i < m; i ++) {
		float new_score = base_score + kDamp * incomming_sums[i];
		error += fabs(new_score - scores_to_test[i]);
		incomming_sums[i] = 0;
	}
	if (error < target_error) printf("Correct\n");
	else printf("Total Error: %f\n", error);
}

