// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "tc.h"
#include <string.h>
#include <timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#define SPMV_VARIANT "ocl_base"

cl_platform_id cpPlatform[32];
cl_uint num_platforms;
cl_device_id device_id;
cl_uint num_devices;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
const char *kernelSource =																"\n" \
"__kernel void tc_kernel(int m, __global int * d_row_offsets, __global int * d_column_indices, __global int * d_degree, int upper, __global int * d_total_sum,	\n" \
"  __global int * d_total)										\n" \
"{														\n" \
"	int src = get_global_id(0);										\n" \
"	//int global_size = get_global_size(0);									\n" \
"	int total_num = 0;											\n" \
"														\n" \
"	if(src < m){								  				\n" \
"		int row_begin_src = d_row_offsets[src];								\n" \
"		int row_end_src = d_row_offsets[src + 1];							  	\n" \
"		for (int offset_src = row_begin_src; offset_src < row_end_src; ++ offset_src){		  	\n" \
"			int dst = d_column_indices[offset_src];						  	\n" \
"			if (dst > src)									  	\n" \
"				break;										\n" \
"			int it = row_begin_src;									\n" \
"			int row_begin_dst = d_row_offsets[dst];						  	\n" \
"			int row_end_dst = d_row_offsets[dst + 1];						  	\n" \
"			for (int offset_dst = row_begin_dst; offset_dst < row_end_dst; ++ offset_dst){		\n" \
"				int dst_dst = d_column_indices[offset_dst];					\n" \
"				if(dst_dst > dst)								\n" \
"					break;									\n" \
"				while (d_column_indices[it] < dst_dst)	  					\n" \
"					it ++;				  					\n" \
"				if (dst_dst == d_column_indices[it])						\n" \
"					total_num ++;			  					\n" \
"			}						  					\n" \
"		}												\n" \
"		atomic_add(d_total, total_num);			 					\n" \
"//	d_total_sum[src] = total_num;										\n" \
"	}													\n" \
"//	barrier(CLK_GLOBAL_MEM_FENCE);													\n" \
"//	    while((src >= 0) && (src < (upper/2))){		  						\n" \
"														\n" \
"//			d_total_sum[src] += d_total_sum[src + upper/2];						\n" \
"//		upper = upper/2;										\n" \
"//		barrier(CLK_GLOBAL_MEM_FENCE);									\n" \
"//	    }													\n" \
"//	barrier(CLK_GLOBAL_MEM_FENCE);													\n" \
"														\n" \
"//	*d_total = d_total_sum[0];											\n" \
"	//*total = total_num;												\n" \
"									 					\n" \
"}														\n" \
														"\n" ;

void TCSolver(int m, int nnz, int *row_offsets, int *column_indices, int *degree, int * total) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(m/(float)localSize)*localSize;
//	printf("\nnumber global is:\t%d\n\n", globalSize);	
	
	int bound_num = 0;	
	while(m > pow(2,bound_num)){
		bound_num ++;
	//printf("\nbound_num in is:\t%d\n\n", bound_num);	
	
	}
	
	printf("\nbound_num is:\t%d\n\n", bound_num);	

	int upper = pow(2,bound_num);
	
	printf("\nupper bound is:\t%d\n\n", upper);	
	
	int *total_sum = (int *)malloc(upper * sizeof(int));
	
	for(int i = 0; i < upper; i++)
		total_sum[i] = 0;

	//Timer t;
	//t.Start();

	//char *filechar = "/home/zy/gardinia_code/src/spmv/base.cl";
	//int sourcesize = 1024*1024;
	//char * source = (char *)calloc(sourcesize, sizeof(char));
	//FILE * fp = fopen(filechar, "rb");
	//fread(source + strlen(source), sourcesize, 1, fp);
	//fclose(fp);
	
	err = clGetPlatformIDs(2, cpPlatform, &num_platforms);
	printf("\nNumber of platforms:\t%u\n\n", num_platforms);	

	//err = clGetPlatformIDs(2, cpPlatform, &num_platforms);
	err = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, NULL, &num_devices);
	if(err < 0){fprintf(stderr, "ERROR get num of devices, err code: %d\n", err); exit(1);}
	//printf("number of devices are %d\n", num_devices);
	
	
	err = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if(err < 0){fprintf(stderr, "ERROR get devices id, err code: %d\n", err); exit(1);}
	//printf("device id is %d\n", device_id);

	//clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES,,);

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	queue = clCreateCommandQueue(context, device_id, 0, &err);
	if(err < 0){fprintf(stderr, "ERROR command queue, err code: %d\n", err); exit(1);}

	//const char * slist[2] = {source, 0};
	//program = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
	if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	//if(err < 0){fprintf(stderr, "ERROR in build program, err code: %d\n", err); exit(1);}

	if(err < 0){
		size_t len;
		char buffer[1000];
	
		printf("error: build program failure\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,1000, buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	kernel = clCreateKernel(program, "tc_kernel", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_degree = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * m, NULL, NULL);
	cl_mem d_total_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * upper, NULL, NULL);
	cl_mem d_total = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), row_offsets, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_degree, CL_TRUE, 0, sizeof(int) * m, degree, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_total_sum, CL_TRUE, 0, sizeof(int) * upper, total_sum, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_total, CL_TRUE, 0, sizeof(int), total, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	err = clSetKernelArg(kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_degree);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &upper);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_total_sum);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_total);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	

	//for(int i = 0; i < 50; i++)
		printf("total before is %d\n", *total);
	
	Timer t;
	t.Start();

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

	//t.Stop();

	clFinish(queue);
	t.Stop();
	err = clEnqueueReadBuffer(queue, d_total, CL_TRUE, 0, sizeof(int), total, 0, NULL, NULL);
//	err|= clEnqueueReadBuffer(queue, d_total_sum, CL_TRUE, 0, sizeof(int)*upper, total_sum, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}

														  				
	//    for(int s = upper >> 1; s > 0; s >>= 1){								  				
	//	for(int j=0; j < s; j++){									  				
	//		total_sum[j] += total_sum[j + s];							  				
	//	}												  					
	//  }												  					

	
//		printf("total_sum in cpu is %d\n", total_sum[0]);
	//*total = total_sum[0];
	printf("total: %d\n", *total);	

	//for(int i = 0; i < m; i++)
	//	printf("%d ", total_sum[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_degree);
	clReleaseMemObject(d_total_sum);
	clReleaseMemObject(d_total);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());

	return;
}

