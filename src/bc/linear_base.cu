// Copyright (c) 2016, Xuhao Chen
#define BC_VARIANT "linear_base"
#include "bc.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "worklistc.h"
#include "timer.h"
#include <vector>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

__global__ void initialize(int m, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) depths[id] = -1;
}

// Shortest path calculation by forward BFS
__global__ void bc_forward(int *row_offsets, int *column_indices, int *path_counts, int *depths, int depth, Worklist2 in_queue, Worklist2 outwl) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if(in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depth)==-1)) {
			//if ((depths[dst] == -1) && (atomicCAS(&depths[dst], -1, depths[src]+1)==-1)) {
				assert(outwl.push(dst));
			}
			if (depths[dst] == depth) {
			//if (depths[dst] == depths[src]+1) {
				atomicAdd(&path_counts[dst], path_counts[src]);
			}
		}
	}
}

// Dependency accumulation by back propagation
__global__ void bc_reverse(int num, int *row_offsets, int *column_indices, int start, int *frontiers, ScoreT *scores, int *path_counts, int *depths, int depth, ScoreT *deltas) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < num) {
		int src = frontiers[start + id];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			//if(depths[dst] == depths[src] + 1) {
			if(depths[dst] == depth + 1) {
				deltas[src] += static_cast<ScoreT>(path_counts[src]) / 
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		scores[src] += deltas[src];
	}
}

void bc_reverse_cpu(int num, int *row_offsets, int *column_indices, int start, int *frontiers, ScoreT *scores, int *path_counts, int *depths, int depth, ScoreT *deltas) {
	for (unsigned id = 0; id < num; id ++) {
		int src = frontiers[start+id];
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src + 1]; 
		for (int offset = row_begin; offset < row_end; offset ++) {
			int dst = column_indices[offset];
			if (depths[dst] == depths[src] + 1) {
				deltas[src] += static_cast<ScoreT>(path_counts[src]) /
					static_cast<ScoreT>(path_counts[dst]) * (1 + deltas[dst]);
			}
		}
		scores[src] += deltas[src];
	}
	return;
}
__global__ void insert(Worklist2 in_queue, int src, int *path_counts, int *depths) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id == 0) {
		in_queue.push(src);
		path_counts[src] = 1;
		depths[src] = 0;
	}
	return;
}

__global__ void push_frontier(Worklist2 in_queue, int *queue, int queue_len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vertex;
	if(in_queue.pop_id(tid, vertex)) {
		queue[queue_len+tid] = vertex;
	}
}

__global__ void bc_normalize(int m, ScoreT *scores, ScoreT max_score) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m) scores[tid] = scores[tid] / (max_score);
}

void BCSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, ScoreT *h_scores) {
	//print_device_info(0);
	int *d_row_offsets, *d_column_indices;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	ScoreT *d_scores, *d_deltas;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_scores, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_deltas, sizeof(ScoreT) * m));
	CUDA_SAFE_CALL(cudaMemset(d_scores, 0, m * sizeof(ScoreT)));
	CUDA_SAFE_CALL(cudaMemset(d_deltas, 0, m * sizeof(ScoreT)));
	int *d_path_counts, *d_depths, *d_frontiers;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_path_counts, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_depths, sizeof(int) * m));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontiers, sizeof(int) * (m+1)));
	CUDA_SAFE_CALL(cudaMemset(d_path_counts, 0, m * sizeof(int)));

	int depth = 0;
	int nitems = 1;
	int frontiers_len = 0;
	vector<int> depth_index;
	depth_index.push_back(0);
	Worklist2 wl1(m), wl2(m);
	Worklist2 *inwl = &wl1, *outwl = &wl2;
	int nthreads = BLOCK_SIZE;
	int nblocks = (m - 1) / nthreads + 1;
	initialize <<<nblocks, nthreads>>> (m, d_depths);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	printf("Launching CUDA BC solver (%d CTAs/SM, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	insert<<<1, 1>>>(*inwl, source, d_path_counts, d_depths);
	do {
		nblocks = (nitems - 1) / nthreads + 1;
		push_frontier<<<nblocks, nthreads>>>(*inwl, d_frontiers, frontiers_len);
		frontiers_len += nitems;
		depth_index.push_back(frontiers_len);
		//printf("Forward: depth=%d, frontire_size=%d\n", depth, nitems);
		depth++;
		bc_forward<<<nblocks, nthreads>>>(d_row_offsets, d_column_indices, d_path_counts, d_depths, depth, *inwl, *outwl);
			//CUDA_SAFE_CALL(cudaMemcpy(h_row_offsets, d_path_counts, sizeof(int) * m, cudaMemcpyDeviceToHost));//zy
			//for(int i=0; i < m; i ++)
			//	printf("%d ", h_row_offsets[i]);//zy
		CudaTest("solving kernel forward failed");
		nitems = outwl->nitems();
		printf("%d\n", nitems);
		Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;
		outwl->reset();
	} while (nitems > 0);
	//printf("\nDone Forward BFS, starting back propagation (dependency accumulation)\n");
		printf("depth_index: %d\n", depth_index.size());

	for(int i=0; i < 34; i ++)
		printf("%d\n", depth_index[i]);//zy
	//CUDA_SAFE_CALL(cudaMemcpy(h_row_offsets, d_path_counts, sizeof(int) * m, cudaMemcpyDeviceToHost));//zy
	//for(int i=0; i < m; i ++)
	//	printf("%d ", h_row_offsets[i]);//zy


	for (int d = depth_index.size() - 2; d >= 0; d--) {
		nitems = depth_index[d+1] - depth_index[d];
		//thrust::sort(thrust::device, d_frontiers+depth_index[d], d_frontiers+depth_index[d+1]);
		nblocks = (nitems - 1) / nthreads + 1;
		//printf("Reverse: depth=%d, frontier_size=%d\n", d, nitems);
		bc_reverse<<<nblocks, nthreads>>>(nitems, d_row_offsets, d_column_indices, depth_index[d], d_frontiers, d_scores, d_path_counts, d_depths, d, d_deltas);
		CudaTest("solving kernel reverse failed");
	}

	
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
//	for(int i=0; i < m; i ++)
//		printf("%f ", h_scores[i]);//zy

	ScoreT *d_max_score;
	d_max_score = thrust::max_element(thrust::device, d_scores, d_scores + m);
	ScoreT h_max_score;
	CUDA_SAFE_CALL(cudaMemcpy(&h_max_score, d_max_score, sizeof(ScoreT), cudaMemcpyDeviceToHost));
		printf("max_score: %f\n", h_max_score);
	nthreads = 512;
	nblocks = (m - 1) / nthreads + 1;
	bc_normalize<<<nblocks, nthreads>>>(m, d_scores, h_max_score);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\titerations = %d.\n", depth);
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_scores, d_scores, sizeof(ScoreT) * m, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_path_counts));
	CUDA_SAFE_CALL(cudaFree(d_depths));
	CUDA_SAFE_CALL(cudaFree(d_deltas));
	CUDA_SAFE_CALL(cudaFree(d_frontiers));
	CUDA_SAFE_CALL(cudaFree(d_row_offsets));
	CUDA_SAFE_CALL(cudaFree(d_column_indices));
}

