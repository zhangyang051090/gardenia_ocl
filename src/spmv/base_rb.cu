// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <stdio.h>
#define SPMV_VARIANT "scalar"
#include "spmv.h"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#include "timer.h"

////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a scalar model (one thread per row)
///////////////////////////////////////////////////////////////////////
//
// spmv_csr_scalar_device
//   Straightforward translation of standard CSR SpMV to CUDA
//   where each thread computes y[i] += A[i,:] * x 
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_csr_scalar_tex_device
//   Same as spmv_csr_scalar_device, except x is accessed via texture cache.
//

__global__ void spmv_csr_scalar_kernel(const int num_rows, const int * Ap,  const int * Aj,
		const ValueType * Ax, const ValueType * x, ValueType * y, float * value, 
		int * block, int * row_start) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < num_rows / 2) {
		float d0 = y[2*row];
		float d1 = y[2*row+1];
		for(int j = row_start[row]; j < row_start[row+1]; j ++){

			d0 += value[j * 4 + 0]*x[2*block[j]+0];
			d0 += value[j * 4 + 1]*x[2*block[j]+1];
			d1 += value[j * 4 + 2]*x[2*block[j]+0];
			d1 += value[j * 4 + 3]*x[2*block[j]+1];
	//		d0 += value[j][1]*x[block[j]+1];
	//		d1 += value[j][2]*x[block[j]+0];
	//		d1 += value[j][3]*x[block[j]+1];
		
		}
		y[2*row] = d0;	
		y[2*row+1] = d1;
	}


}
/*
__global__ void spmv_csr_scalar_kernel(const int num_rows, const int * Ap,  const int * Aj,
		const ValueType * Ax, const ValueType * x, ValueType * y) {
;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < num_rows) {
		for(int i = 0; i < num_rows; i ++)
			B[i*num_rows + row] = 0.0;
	}

	if(row < num_rows) {
		for(int j = 0; j < num_rows; j ++)	
			for(int offset = Ap[row]; offset < Ap[row+1]; offset ++)
				B[j*num_rows + Aj[offset]] = Ax[offset];
	}

	//for (int i = 0; i < num_rows*num_rows; i=i+4){
	if((row%4 == 0) && row < num_rows){
			sum0 += B[row] * x[row];			
			sum1 += B[row+num_rows] * x[row];
			sum0 += B[row] * x[row + 1];			
			sum1 += B[row+num_rows] * x[row+1];
	}
	
	sum0= atomic(sum0);
	sum1= atomic(sum1);
	if(row == 0)
		sum = sum0 + sum1;

	int block_total = BlockReduce(temp_storage).Sum(local_total);
	if(threadIdx.x == 0) atomicAdd(total, block_total);


	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int bm = 2*row;
	//if(((bm % 2) == 0) && (bm < num_rows)) {
	if(bm < num_rows) {
	  //   for(int i = 0; i < 2; i ++){
	//	ValueType sum = y[bm+i];
		ValueType sum = y[bm];
			colidx[nnz/2];
		b_Ap[bm] = Ap[bm] / 2;

		int row_begin_1 = Ap[bm];
		int row_end_1 = Ap[bm+1];
		for (int offset = row_begin_1; offset < row_end_1; offset ++){
			colidx[offset] = Aj[offset]/2;
			if(offset > 0 && colidx[offset]!=colidx[offset-1])
				block_size++;
		}

		int row_begin_2 = Ap[bm+1];
		int row_end_2 = Ap[bm+2];
		for (int offset = row_begin_2; offset < row_end_2; offset ++){
			colidx[offset] = Aj[offset]/2;
			if(offset > 0 && (colidx[offset]!=colidx[offset-1]) &&)
				block_size++;
		}	

		b_Ax[offset] = 0;
		b_Ax[offset] = Ax[offset];

		


		//row_begin = min(row_begin_1, row_begin_2)/2;
		//row_end = max(row_end_1, row_end_2)/2;

		for (int bm_offset = row_begin_1; bm_offset < row_end_2; bm_offset ++){
		for (int block = 0; block < num_rows; block ++){
			//block spmv
			//for(int offset = bm_offset; offset < bm_offset + 4; offset ++){
		   
			




			for(int i = 0; i < 4; i ++)
				bitmask[i]=0;

			if(Aj[bm_offset]%2==0 && row%2==0)bitmask[0]=1;
			else if(Aj[bm_offset]%2==1 && row%2==0)bitmask[1]=1;
			else if(Aj[bm_offset]%2==0 && row%2==1)bitmask[2]=1;
			else bitmask[3]=1;	
			
			sum0 += Ax[bm_offset]*bitmask[0] * x[2*colidx[bm_offset]];			
			sum1 += Ax[bm_offset+(bm_row_end - bm_row_begin)]*bitmask[1] * x[2*colidx[bm_offset]];
			sum0 += Ax[bm_offset]*bitmask[2] * x[2*colidx[bm_offset] + 1];			
			sum1 += Ax[bm_offset+(bm_row_end - bm_row_begin)]*bitmask[3] * x[2*colidx[bm_offset] + 1];			
		}
	//	y[bm+i] = sum;
		y[bm] = sum;
	  //    }
	}

	int bm = blockIdx.x * blockDim.x + threadIdx.x;
	//if(((bm % 2) == 0) && (bm < num_rows)) {
	if(bm < num_rows) {
	  //   for(int i = 0; i < 2; i ++){
	//	ValueType sum = y[bm+i];
		ValueType sum = y[bm];
		b_Ap[bm] = Ap[bm] / 2;
		int row_begin = Ap[bm];
		int row_end = Ap[bm+1];
		bm_row_begin = row_begin / 2;
		bm_row_end = row_end / 2;
		for (int bm_offset = bm_row_begin; bm_offset < bm_row_end; bm_offset ++){
			//block spmv
			//for(int offset = bm_offset; offset < bm_offset + 4; offset ++){			
			sum0 += Ax[bm_offset] * x[Aj[offset]];			
			sum1 += Ax[bm_offset+(bm_row_end - bm_row_begin)] * x[Aj[offset]];
			sum0 += Ax[bm_offset] * x[Aj[offset] + 1];			
			sum1 += Ax[bm_offset+(bm_row_end - bm_row_begin)] * x[Aj[offset] + 1];			
		}
	//	y[bm+i] = sum;
		y[bm] = sum;
	  //    }
	}



}
*/
void SpmvSolver(int num_rows, int nnz, int *h_Ap, int *h_Aj, ValueType *h_Ax, ValueType *h_x, ValueType *h_y, ValueType *h_value, int *h_block, int *h_row_start, int &num_block_all) { 
	//print_device_info(0);
	int *d_Ap, *d_Aj;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ap, (num_rows + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Aj, nnz * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ap, h_Ap, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_Aj, h_Aj, nnz * sizeof(int), cudaMemcpyHostToDevice));
	ValueType *d_Ax, *d_x, *d_y;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_Ax, sizeof(ValueType) * nnz));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(ValueType) * num_rows));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(ValueType) * num_rows));
	CUDA_SAFE_CALL(cudaMemcpy(d_Ax, h_Ax, nnz * sizeof(ValueType), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, num_rows * sizeof(ValueType), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, num_rows * sizeof(ValueType), cudaMemcpyHostToDevice));
	float *d_value;
	int *d_block, *d_row_start;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_value, num_block_all * 4 * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_block, num_block_all * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_start, (num_rows / 2 + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_value, h_value, num_block_all * 4 * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_block, h_block, num_block_all * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_row_start, h_row_start, (num_rows / 2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
	
	for(int j = 0; j < 300; j ++){
			printf("h_value: %f \n", h_value[j]);
	}


	int nthreads = BLOCK_SIZE;
	int nblocks = (num_rows - 1) / nthreads + 1;
	printf("Launching CUDA SpMV solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

	Timer t;
	t.Start();
	spmv_csr_scalar_kernel <<<nblocks, nthreads>>> (num_rows, d_Ap, d_Aj, d_Ax, d_x, d_y, d_value, d_block, 
		d_row_start);   
	CudaTest("solving failed");
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	t.Stop();

	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	CUDA_SAFE_CALL(cudaMemcpy(h_y, d_y, sizeof(ValueType) * num_rows, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaFree(d_Ap));
	CUDA_SAFE_CALL(cudaFree(d_Aj));
	CUDA_SAFE_CALL(cudaFree(d_Ax));
	CUDA_SAFE_CALL(cudaFree(d_x));
	CUDA_SAFE_CALL(cudaFree(d_y));
	CUDA_SAFE_CALL(cudaFree(d_value));
	CUDA_SAFE_CALL(cudaFree(d_block));
	CUDA_SAFE_CALL(cudaFree(d_row_start));
}

