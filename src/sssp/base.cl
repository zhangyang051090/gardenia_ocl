// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>

#pragma OPENCL EXTENSION cl_khr_int32_base_atomics: enable
__kernel void insert(int source, __global int *d_nitems, __global int *d_in_frontier)
{																			
																			
	int tid = get_global_id(0);
	if( tid == 0 ){	
		d_in_frontier[*d_nitems] = source;
		(*d_nitems) ++;
	}
		return;												
}																			
																			



__kernel void sssp_kernel(int m, __global int *d_nitems, __global int *d_out_items, __global int * d_row_offsets, __global int * d_column_indices, __global unsigned *weight,  __global unsigned * d_dist, __global int *d_in_frontier, __global int *d_out_frontier)
{																																					
	int tid = get_global_id(0);															

	int src, tmp_1;
// for(int tid = 0; tid < m; tid ++){
//int tid = 0;
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
				unsigned old_dist = d_dist[dst];											
				unsigned new_dist = d_dist[src] + weight[offset];											
				barrier(CLK_GLOBAL_MEM_FENCE);
				if(new_dist < old_dist){
					
					if(atomic_min(&d_dist[dst], new_dist) > new_dist){
						//d_dist[dst] = new_dist;
						int lindex = atomic_add(d_out_items, 1);
						d_out_frontier[lindex] = dst;

						//d_out_frontier[*d_out_items] = dst;
						//*d_out_items = *d_out_items + 1;
					}
				}										
			}																
		}																	
// }															
	
}

