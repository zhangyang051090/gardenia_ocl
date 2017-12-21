__kernel void spmv_kernel(int num_rows, __global int * Ap, __global int * Aj, __global ValueType * Ax,  __global ValueType * x, __global ValueType * y) {
	int idx = get_global_id(0);
	if(idx < num_rows){
	y[idx] = 5.0;	

	}
	return;
}
