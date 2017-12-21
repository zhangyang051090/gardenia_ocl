// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <vector>
#include <string.h>
#include "common.h"
#include "sssp.h"
//#include "worklisto.h"
#include <timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#define SSSP_VARIANT "ocl_base"

cl_platform_id cpPlatform[32];
cl_uint num_platforms;
cl_device_id device_id[5];
cl_uint num_devices;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel sssp_kernel;
cl_kernel insert;

void SSSPSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, DistT *h_weight, DistT *h_dist, int delta) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 16;
	globalSize = ceil(m/(float)localSize)*localSize;

	char *filechar = "/home/zy/gardinia_code/src/sssp/base.cl";
	int sourcesize = 1024*1024;
	char * source_1 = (char *)calloc(sourcesize, sizeof(char));
	FILE * fp = fopen(filechar, "rb");
	fread(source_1 + strlen(source_1), sourcesize, 1, fp);
	fclose(fp);
	
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	printf("\nNumber of platforms:\t%u\n\n", num_platforms);	

	err = clGetPlatformIDs(num_platforms, cpPlatform, NULL);
	printf("\nplatform id:\t%u\n\n", cpPlatform[0]);	
	printf("\nplatform id:\t%u\n\n", cpPlatform[1]);	
	
	size_t size;
	size_t maxComputeUnits;
	err = clGetPlatformInfo(cpPlatform[0], CL_PLATFORM_NAME, 0, NULL, &size);
	char * name_p = (char *)alloca(sizeof(char) * size);
	err = clGetPlatformInfo(cpPlatform[0], CL_PLATFORM_NAME, size, name_p, NULL);
	printf("\nplatform name:\t%s\n\n", name_p);


	//err = clGetPlatformIDs(2, cpPlatform, &num_platforms);
	err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &num_devices);
	if(err < 0){fprintf(stderr, "ERROR get num of devices, err code: %d\n", err); exit(1);}
	printf("number of devices are %d\n", num_devices);
	
	
	err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id[0], NULL);
	if(err < 0){fprintf(stderr, "ERROR get devices id, err code: %d\n", err); exit(1);}
	printf("device id is %d\n", device_id[0]);

	err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id[1], NULL);
	if(err < 0){fprintf(stderr, "ERROR get devices id, err code: %d\n", err); exit(1);}
	printf("device id is %d\n", device_id[1]);

	err = clGetDeviceInfo(device_id[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, &size);
	if(err < 0){fprintf(stderr, "ERROR get devices info, err code: %d\n", err); exit(1);}
	printf("max compute units is %d\n", maxComputeUnits);
	
	
	err = clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, 0, NULL, &size);
	char * name = (char *)alloca(sizeof(char) * size);
	err = clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, size*sizeof(char), name, NULL);
	printf("\ndev name:\t%s\n\n", name);
	
	context = clCreateContext(0, 1, &device_id[0], NULL, NULL, &err);
	queue = clCreateCommandQueue(context, device_id[0], 0, &err);
	if(err < 0){fprintf(stderr, "ERROR command queue, err code: %d\n", err); exit(1);}

	const char * slist[2] = {source_1, 0};
	program = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if(err < 0){
		size_t len;
		char buffer[1000];
	
		printf("error: build program failure\n");
		clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,1000, buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}
	
	sssp_kernel = clCreateKernel(program, "sssp_kernel", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	insert = clCreateKernel(program, "insert", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	int *in_frontier = (int *) malloc(m * sizeof(int));
	int *out_frontier = (int *) malloc(m * sizeof(int));
	
	int out_items = 0, nitems = 0;

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_weight = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(DistT) * nnz, NULL, NULL);
	cl_mem d_dist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(DistT) * m, NULL, NULL);
	cl_mem d_nitems = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_in_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_out_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	//int d_in_items = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_out_items = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	DistT zero = 0;
	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), h_row_offsets , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, h_column_indices, 0, NULL, NULL);
	//DistT * d_dist;
	err |= clEnqueueWriteBuffer(queue, d_weight, CL_TRUE, 0, sizeof(DistT) * nnz, h_weight, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_dist, CL_TRUE, 0, sizeof(DistT) * m, h_dist, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_dist, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);//zy: with some doubts here.
	err |= clEnqueueWriteBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	err = clSetKernelArg(insert, 0, sizeof(int), &source);
	err |= clSetKernelArg(insert, 1, sizeof(cl_mem), &d_nitems);
	err |= clSetKernelArg(insert, 2, sizeof(cl_mem), &d_in_frontier);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}


		
	err = clSetKernelArg(sssp_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(sssp_kernel, 1, sizeof(cl_mem), &d_nitems);
	err |= clSetKernelArg(sssp_kernel, 2, sizeof(cl_mem), &d_out_items);
	err |= clSetKernelArg(sssp_kernel, 3, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(sssp_kernel, 4, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(sssp_kernel, 5, sizeof(cl_mem), &d_weight);
	err |= clSetKernelArg(sssp_kernel, 6, sizeof(cl_mem), &d_dist);
	err |= clSetKernelArg(sssp_kernel, 7, sizeof(cl_mem), &d_in_frontier);
	err |= clSetKernelArg(sssp_kernel, 8, sizeof(cl_mem), &d_out_frontier);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}


	int iter=0;

	Timer t;
	t.Start();

	err = clEnqueueNDRangeKernel(queue, insert, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	//nitems = in_frontier->nitems();

	err = clEnqueueReadBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 1, err code: %d\n", err); exit(1);}
	
		printf("zyy-nitems %d \n", nitems);
		printf("zyy-row %d \n", h_row_offsets[0]);
		printf("zyy-row1 %d \n", h_row_offsets[1]);

	do {
		++iter;
	
	//	printf("iteration start....\n");
	
		err = clEnqueueReadBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 1, err code: %d\n", err); exit(1);}
		
		if(nitems < 64)globalSize = 64;
		else if(nitems < 128)globalSize = 128;
		else if(nitems < 256)globalSize = 256;
		else if(nitems < 512)globalSize = 512;
		else if(nitems < 1024)globalSize = 1024;
		else if(nitems < 2048)globalSize = 2048;
		else if(nitems < 4096)globalSize = 4096;
		else if(nitems < 8192)globalSize = 8192;
		else if(nitems < 16384)globalSize = 16384;
		else if(nitems < 32768)globalSize = 32768;
		else globalSize = ceil(m/(float)localSize)*localSize;
	
		err = clEnqueueNDRangeKernel(queue, sssp_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
			
		err = clEnqueueReadBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}
		
		printf("zyy-out_items %d \n", out_items);
		
		//err = clEnqueueReadBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
		//if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 4, err code: %d\n", err); exit(1);}
		
		//err = clEnqueueReadBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
		//if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 5, err code: %d\n", err); exit(1);}
		
		//int *tmp = in_frontier;
		//in_frontier = out_frontier;
		//out_frontier = tmp;


		
		//err = clEnqueueWriteBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
		//if(err < 0){fprintf(stderr, "ERROR write buffer 3, err code: %d\n", err); exit(1);}
		//err = clEnqueueWriteBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
		//if(err < 0){fprintf(stderr, "ERROR write buffer 4, err code: %d\n", err); exit(1);}
		
		err = clEnqueueCopyBuffer(queue, d_in_frontier, d_tmp, 0, 0, nitems * sizeof(int), 0, NULL, NULL);		
		if(err < 0){fprintf(stderr, "ERROR copy buffer, err code: %d\n", err); exit(1);}

		err = clEnqueueCopyBuffer(queue, d_out_frontier, d_in_frontier, 0, 0, out_items * sizeof(int), 0, NULL, NULL);		
		if(err < 0){fprintf(stderr, "ERROR copy buffer, err code: %d\n", err); exit(1);}

		err = clEnqueueCopyBuffer(queue, d_tmp, d_out_frontier, 0, 0, nitems * sizeof(int), 0, NULL, NULL);		
		if(err < 0){fprintf(stderr, "ERROR copy buffer, err code: %d\n", err); exit(1);}
		
		err = clEnqueueWriteBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 5, err code: %d\n", err); exit(1);}
		
		err = clEnqueueWriteBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 6, err code: %d\n", err); exit(1);}


	//printf("\tzyy-runtime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	//	printf("end of one iteration ...\n");
	} while(out_items > 0);
	//t.Stop();

	clFinish(queue);
	
	t.Stop();
	
	err = clEnqueueReadBuffer(queue, d_dist, CL_TRUE, 0, sizeof(DistT) * m, h_dist , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}

	

	
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_weight);
	clReleaseMemObject(d_dist);
	clReleaseMemObject(d_nitems);
	clReleaseMemObject(d_out_items);
	clReleaseMemObject(d_tmp);
	clReleaseMemObject(d_in_frontier);
	clReleaseMemObject(d_out_frontier);
	free(in_frontier);
	free(out_frontier);
	clReleaseProgram(program);
	clReleaseKernel(insert);
	clReleaseKernel(sssp_kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, t.Millisecs());

	return;
}

