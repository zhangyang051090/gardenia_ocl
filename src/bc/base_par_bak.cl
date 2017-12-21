// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>


__kernel void initialize(int m, __global int *d_depths)
{																			
																			
	int tid = get_global_id(0);
	//for(int tid = 0; tid < m; tid ++){
	if( tid < m ){	
		d_depths[tid] = -1;
	}
	
}																			
																			



__kernel void bc_forward(int m, __global int *d_nitems, __global int *d_out_items, __global int * d_row_offsets, __global int * d_column_indices, __global int * d_path_counts, __global int *d_depths, int depth, __global int *d_in_frontier, __global int *d_out_frontier)
{																																					
	int tid = get_global_id(0);															
	int src, tmp_1;
// for(int tid = 0; tid < m; tid ++){
	if(tid < (*d_nitems)){
		src = d_in_frontier[tid];																		
		tmp_1 = 1;
	}
	else tmp_1 = 0;
	
	if(tmp_1){	
			int row_begin = d_row_offsets[src];												
			int row_end = d_row_offsets[src + 1];												
			for (int offset = row_begin; offset < row_end; ++ offset){									
				int dst = d_column_indices[offset];
				//unsigned new_dist = d_dist[src] + 1;											
				if(d_depths[dst] == -1){
					d_depths[dst] = depth;
			
					int lindex = atomic_add(d_out_items, 1);
					d_out_frontier[lindex] = dst;
					//barrier(CLK_GLOBAL_MEM_FENCE);
					//*d_out_items = *d_out_items + 1;
					//atomic_add(d_out_items, 1);
				}	

					barrier(CLK_GLOBAL_MEM_FENCE);
				if(d_depths[dst] == depth){
				//d_path_counts[dst] = d_path_counts[dst] + d_path_counts[src];
				atomic_add(&d_path_counts[dst], d_path_counts[src]);
				}									
			}																
		}																	
															
	
}


__kernel void bc_reverse( __global int *d_nitems, __global int * d_row_offsets, __global int * d_column_indices, int start, __global float * d_scores, __global int* d_path_counts, __global int* d_depths, int depth, __global float *d_deltas, __global int *d_in_frontier)
{																																					
	int tid = get_global_id(0);															

	int src, tmp_1;
// for(int tid = 0; tid < 875713; tid ++){


	if(tid < (*d_nitems)){																
			int src = d_in_frontier[start + tid];
			int row_begin = d_row_offsets[src];												
			int row_end = d_row_offsets[src + 1];												
			for (int offset = row_begin; offset < row_end; ++ offset){									
				int dst = d_column_indices[offset];
				if(d_depths[dst] == depth + 1){
					d_deltas[src] += (1.0 + d_deltas[dst]) * (d_path_counts[src])/(d_path_counts[dst]);

				}										
			}
		//	atomic_add(&d_scores[src], d_deltas[src]);																
			d_scores[src] += d_deltas[src];
		}																	
// }															
	
}


__kernel void insert(int source, __global int *d_nitems, __global int *d_path_counts, __global int *d_depths, __global int *d_in_frontier)
{																			
																			
	int tid = get_global_id(0);
	//int tid = 0;
	if( tid == 0 ){	
		d_in_frontier[*d_nitems] = source;
		(*d_nitems) ++;
		d_path_counts[source] = 1;
		d_depths[source] = 0;
	}
}																			

__kernel void push_frontier(__global int* d_nitems, __global int *d_in_frontier, __global int *d_frontier, int frontiers_len)
{																			
																			
	int tid = get_global_id(0);
	//for (int tid = 0; tid < 875713; tid ++){
	int vertex;
	int tmp_1;
	if(tid < (*d_nitems)){
		vertex = d_in_frontier[tid];
		tmp_1 = 1;
	}
	else tmp_1 = 0;
	
	if( tmp_1 == 1 ){	
		d_frontier[frontiers_len + tid] = vertex;
	}

}												

__kernel void bc_normalize(int m, __global float *d_scores, __global float *d_max_score)
{	
	//for (int tid = 0; tid < m; tid ++){																							
	int tid = get_global_id(0);
	if( tid < m ){	
		d_scores[tid] = d_scores[tid] / (*d_max_score);
	}
	
}


__kernel void max_element(int m, __global float *d_scores, __global float *d_max_score)
{																								

//	float tmp = d_scores[0];
//	for (int i = 0; i < m; i++){
//			if(tmp < d_scores[i]){
//				tmp = d_scores[i];
//			}
//			
//	}
//	*d_max_score = tmp;
	

	int tid = get_global_id(0);
	__local tmp;	
	int j = 0;
	if(tid == 0){
		while((2<<j) < m){
			j++;
		}
		tmp = (2<<j);
		
		for(int i = m; i < tmp; i++)
			{
				d_scores[i] = 0;
			}
	}
	
		while((tid >=0) && (tid < (tmp/2))){	
			d_scores[tid] = (d_scores[tid] > d_scores[tid + tmp/2])? d_scores[tid]:d_scores[tid + tmp/2];
			tmp = tmp/2;
		}
	*d_max_score = d_scores[0];
}
