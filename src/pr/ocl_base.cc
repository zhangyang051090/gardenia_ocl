// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <vector>
#include <string.h>
#include "common.h"
#include "pr.h"
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
//cl_program program_gather;
cl_kernel calc_contrib;
cl_kernel gather;
cl_kernel reduction;
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics:enable
void PRSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *degree, ScoreT *h_scores) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(m/(float)localSize)*localSize;
	
	int k = 0;
	int q;
	while((2 << k) < m){
		k++;
	}

	q = (2<<k);
	printf("m: %d, q: %d\n",m, q);
	float *h_reduce = (float *) malloc(q * sizeof(float));

	for(int i = 0; i < q; i ++)h_reduce[i] = 0.0;

	//Timer t;
	//t.Start();

	char *filechar = "/home/zy/gardinia_code/src/pr/base.cl";
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));
	FILE * fp = fopen(filechar, "rb");
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
	
	err = clGetPlatformIDs(2, cpPlatform, &num_platforms);
	printf("\nNumber of platforms:\t%u\n\n", num_platforms);	

	//err = clGetPlatformIDs(2, cpPlatform, &num_platforms);
	err = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, NULL, &num_devices);
	if(err < 0){fprintf(stderr, "ERROR get num of devices, err code: %d\n", err); exit(1);}
	printf("number of devices are %d\n", num_devices);
	
	
	err = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if(err < 0){fprintf(stderr, "ERROR get devices id, err code: %d\n", err); exit(1);}
	printf("device id is %d\n", device_id);


	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	queue = clCreateCommandQueue(context, device_id, 0, &err);
	if(err < 0){fprintf(stderr, "ERROR command queue, err code: %d\n", err); exit(1);}

	const char * slist[2] = {source, 0};
	program = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	//program_calc_contrib = clCreateProgramWithSource(context, 1, (const char **) & kernelSource_calc_contrib, NULL, &err);
	if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}
	
	//program_gather = clCreateProgramWithSource(context, 1, (const char **) & kernelSource_gather, NULL, &err);
	//if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}

	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	//err = clBuildProgram(program, 0, NULL, "-I /home/zy/gardinia_code/include/common.h", NULL, NULL);
	//if(err < 0){fprintf(stderr, "ERROR in build program, err code: %d\n", err); exit(1);}

	if(err < 0){
		size_t len;
		char buffer[1000];
	
		printf("error: build program failure\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,1000, buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}
	
	
	//err = clBuildProgram(program_gather, 0, NULL, NULL, NULL, NULL);
	//err = clBuildProgram(program, 0, NULL, "-I /home/zy/gardinia_code/include/common.h", NULL, NULL);
	//if(err < 0){fprintf(stderr, "ERROR in build program, err code: %d\n", err); exit(1);}

	/*if(err < 0){
		size_t len;
		char buffer[1000];
	
		printf("error: build program failure\n");
		clGetProgramBuildInfo(program_gather, device_id, CL_PROGRAM_BUILD_LOG,1000, buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}*/
	
	calc_contrib = clCreateKernel(program, "calc_contrib", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	gather = clCreateKernel(program, "gather", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	reduction = clCreateKernel(program, "reduction", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_reduce = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * q, NULL, NULL);
	cl_mem d_outgoing_contrib = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	cl_mem d_degree = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT) * m, NULL, NULL);
	cl_mem d_diff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), in_row_offsets , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, in_column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_scores, CL_TRUE, 0, sizeof(ScoreT) * m, h_scores, 0, NULL, NULL);
	//err |= clEnqueueWriteBuffer(queue, d_reduce, CL_TRUE, 0, sizeof(float) * q, h_reduce, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_degree, CL_TRUE, 0, sizeof(int) * m, degree, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	err = clSetKernelArg(calc_contrib, 0, sizeof(int), &m);
	//err |= clSetKernelArg(calc_contrib, 1, sizeof(int), &nnz);
	//err |= clSetKernelArg(calc_contrib, 2, sizeof(cl_mem), &d_row_offsets);
	//err |= clSetKernelArg(calc_contrib, 3, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(calc_contrib, 1, sizeof(cl_mem), &d_outgoing_contrib);
	err |= clSetKernelArg(calc_contrib, 2, sizeof(cl_mem), &d_degree);
	err |= clSetKernelArg(calc_contrib, 3, sizeof(cl_mem), &d_scores);
	//err |= clSetKernelArg(calc_contrib, 7, sizeof(cl_mem), &d_diff);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}


		
	err = clSetKernelArg(gather, 0, sizeof(int), &m);
	err |= clSetKernelArg(gather, 1, sizeof(int), &nnz);
	err |= clSetKernelArg(gather, 2, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(gather, 3, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(gather, 4, sizeof(cl_mem), &d_reduce);
	err |= clSetKernelArg(gather, 5, sizeof(int), &q);
	err |= clSetKernelArg(gather, 6, sizeof(cl_mem), &d_outgoing_contrib);
	err |= clSetKernelArg(gather, 7, sizeof(cl_mem), &d_degree);
	err |= clSetKernelArg(gather, 8, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(gather, 9, sizeof(cl_mem), &d_diff);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
        //clFinish(queue);

	err = clSetKernelArg(reduction, 0, sizeof(cl_mem), &d_reduce);
	err |= clSetKernelArg(reduction, 1, sizeof(int), &q);
	
	//for(int i = 0; i < 50; i++)
	//	printf("value of y-bef is %f\n", y[i]);


	int iter=0;
	float diff;	
	Timer t;
	t.Start();

	do {
		++iter;
		diff = 0;
		int d = q;		
	
		err = clEnqueueNDRangeKernel(queue, calc_contrib, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	
	
		err = clEnqueueWriteBuffer(queue, d_diff, CL_TRUE, 0, sizeof(float), &diff, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}
	
		clEnqueueWriteBuffer(queue, d_reduce, CL_TRUE, 0, sizeof(float) * q, h_reduce, 0, NULL, NULL);
		err = clEnqueueNDRangeKernel(queue, gather, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	
		while(d > 1){	
		
			clSetKernelArg(reduction, 1, sizeof(int), &d);
			clEnqueueNDRangeKernel(queue, reduction, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		
			d = d/2;
		}
		err = clEnqueueReadBuffer(queue, d_reduce, CL_TRUE, 0, sizeof(float), &diff, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}
		
		printf("2 %2d   %lf\n", iter, diff);
	} while(diff > EPSILON && iter < MAX_ITER);

	clFinish(queue);
	
	t.Stop();
	
	err = clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, sizeof(ScoreT) * m, h_scores , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}

	

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-after is %f\n", y[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_reduce);
	clReleaseMemObject(d_outgoing_contrib);
	clReleaseMemObject(d_degree);
	clReleaseMemObject(d_scores);
	clReleaseMemObject(d_diff);
	clReleaseProgram(program);
	//clReleaseProgram(program_gather);
	clReleaseKernel(calc_contrib);
	clReleaseKernel(gather);
	clReleaseKernel(reduction);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(h_reduce);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());

	return;
}

