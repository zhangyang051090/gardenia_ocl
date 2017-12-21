// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
//#pragma OPENCL EXTENTION cl_khr_int64_base_atomics : enable
inline void AtomicAdd(volatile  __global float * diff, const float error){  			
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
}

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

__kernel void calc_contrib(int m, __global float * outgoing_contrib, __global int * degree, __global float * scores)
{																			
																			
	int tid = get_global_id(0);															
																			
		if( tid < m )																
			outgoing_contrib[tid] = scores[tid] / degree[tid];										
}																			
																			



__kernel void gather(int m, int nnz, __global int * row_offsets, __global int * column_indices, __global float *d_reduce, int q,  __global float * outgoing_contrib, __global int * degree, __global float * scores, __global float * diff)
{																			
																			
	int src = get_global_id(0);															
																			
	const float base_score = (1.0f - 0.85) / m;													
	//float *tmp;
		float error = 0.0;															
		if( src < m ){																
			float incoming_total = 0;													
			int row_begin = row_offsets[src];												
			int row_end = row_offsets[src + 1];												
			for (int offset = row_begin; offset < row_end; offset ++){									
				int dst = column_indices[offset];											
				incoming_total += outgoing_contrib[dst];										
			}																
			float old_score = scores[src];													
			scores[src] = base_score + 0.85 * incoming_total;										
			error += fabs(scores[src] - old_score);												
		//	AtomicAdd(diff, error);
			d_reduce[src] = error;
		}
//		barrier(CLK_GLOBAL_MEM_FENCE);	
//		while((src >=0) && (src < (q/2))){
//			d_reduce[src] = d_reduce[src] + d_reduce[src + (q/2)];
//		barrier(CLK_GLOBAL_MEM_FENCE);	
//			q = q/2;
//		}																
		//barrier(CLK_GLOBAL_MEM_FENCE);	
//		if(src == 0)
//		*diff = d_reduce[0] + (*diff);
		//AtomicAdd(diff, d_reduce[0]);

	return;	
}

__kernel void reduction(__global float *d_reduce, int q){
	int src = get_global_id(0);
	
	if((src >=0) && (src < (q/2))){
		d_reduce[src] = d_reduce[src] + d_reduce[src + (q/2)];
	}



}																	
