// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>

/*inline void AtomicAdd(volatile __global float * diff, const float error){  			
		union{																	
			unsigned int intVal;														
			float floatVal;															
		}newVal;																
																			
		union{																	
			unsigned int intVal;														
			float floatVal;															
		}prevVal;																
			
			do{																
				prevVal.floatVal = *diff;												
				newVal.floatVal =prevVal.floatVal + error;										
			}while(atomic_cmpxchg((volatile __global unsigned int *)diff,									
						prevVal.intVal, newVal.intVal)!=prevVal.intVal);							
}*/

/*inline float atomic_add_float(__global float* const address,
					      const float value){

	uint oldval, newval, readback;

	*(float*)&oldval = *address;
	*(float*)&newval = (*(float*)&oldval + value);
	while((readback = atomic_cmpxchg((__global uint*)address, oldval, newval)) != oldval){
		oldval = readback;
		*(float*)&newval = (*(float*)&oldval + value);																	
	}
	return *(float*)&oldval;
}*/

__kernel void insert(int source, __global int *d_nitems, __global int *d_in_frontier)
{																			
																			
	int tid = get_global_id(0);
	//int j;															
	if( tid == 0 ){	
		//queue.push_back(source);
		d_in_frontier[*d_nitems] = source;
		(*d_nitems) ++;
	}
	

		//int lindex = atomicAdd((int *) d_index, 1);
		//if(lindex < *d_size)
		//	queue[lindex] = item;
	//int d_nitems = 0;
	//d_nitems = queue.size();

		return;												
}																			
																			



__kernel void bfs_kernel(int m, __global int *d_nitems, __global int *d_out_items, __global int * d_row_offsets, __global int * d_column_indices, __global float * d_dist, __global int *d_in_frontier, __global int *d_out_frontier)
{																																					
//	int tid = get_global_id(0);															

for(int tid = 0; tid < m; tid ++){
	int tid = 0;
	int src, tmp_1, tmp_2;

	if(tid < (*d_nitems)){
		src = d_in_frontier[tid];																		
		tmp_1 = 1;
	}
	else tmp_1 = 0;
  		
		if( tmp_1){																
			int row_begin = d_row_offsets[src];												
			int row_end = d_row_offsets[src + 1];												
			
				(*d_out_items) = row_end;
			for (int offset = row_begin; offset < row_end; offset ++){									
				int dst = d_column_indices[offset];
				float new_dist = d_dist[src] + 1;											
				if(d_dist[dst] == 1000000000){
					d_dist[dst] = new_dist;
					//assert(out_queue.push(dst));
					
					//tmp_2 = *d_out_items;
					d_out_frontier[*d_out_items] = dst;
					(*d_out_items)++;
					//int lindex = atomicAdd((int *) d_index, 1);
					//if(lindex >= *d_size)
					//	tmp_2 =  0;
					//out_queue[lindex] = dst;
					//	tmp_2 =  1;
					//assert(tmp_2);
				}										
			}																
		}																	
}															
	
}


																
//__kernel void insert(int source, __global *d_nitems __global * queue){}



//__kernel void bfs_kernel(int m, __global int * row_offsets, __global int * column_indices, __global float * dist)
//{}
