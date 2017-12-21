// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <vector>
#include <string.h>
#include "common.h"
#include "bfs.h"
#include <timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#define BFS_VARIANT "ocl_base"

cl_platform_id cpPlatform[32];
cl_uint num_platforms;
cl_device_id device_id;
cl_uint num_devices;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel bfs_kernel;
cl_kernel insert;
//cl_kernel exchange;

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, DistT *h_dist) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize =64;
	globalSize = ceil(m/(float)localSize)*localSize;
	
	char *filechar = "/home/zy/gardinia_code/src/bfs/base.cl";
	int sourcesize = 1024*1024;
	char * source_1 = (char *)calloc(sourcesize, sizeof(char));
	FILE * fp = fopen(filechar, "rb");
	fread(source_1 + strlen(source_1), sourcesize, 1, fp);
	fclose(fp);
	
	err = clGetPlatformIDs(2, cpPlatform, &num_platforms);
	printf("\nNumber of platforms:\t%u\n\n", num_platforms);	

	err = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, NULL, &num_devices);
	if(err < 0){fprintf(stderr, "ERROR get num of devices, err code: %d\n", err); exit(1);}
	printf("number of devices are %d\n", num_devices);
	
	
	err = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if(err < 0){fprintf(stderr, "ERROR get devices id, err code: %d\n", err); exit(1);}
	printf("device id is %d\n", device_id);


	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	queue = clCreateCommandQueue(context, device_id, 0, &err);
	if(err < 0){fprintf(stderr, "ERROR command queue, err code: %d\n", err); exit(1);}

	const char * slist[2] = {source_1, 0};
	program = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}
	
	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	if(err < 0){
		size_t len;
		char buffer[1000];
	
		printf("error: build program failure\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,1000, buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}
	
	
	bfs_kernel = clCreateKernel(program, "bfs_kernel", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	insert = clCreateKernel(program, "insert", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	//exchange = clCreateKernel(program, "exchange", &err);
	//if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}
	
	int *in_frontier = (int *) malloc(m * sizeof(int));
	int *out_frontier = (int *) malloc(m * sizeof(int));
	
	int out_items = 0, nitems = 0;

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_dist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(DistT) * m, NULL, NULL);
	cl_mem d_nitems = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_in_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_out_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_out_items = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);

	int zero = 0;
	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), h_row_offsets , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, h_column_indices, 0, NULL, NULL);
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


		
	err = clSetKernelArg(bfs_kernel, 0, sizeof(int), &m);
	err |= clSetKernelArg(bfs_kernel, 1, sizeof(cl_mem), &d_nitems);
	err |= clSetKernelArg(bfs_kernel, 2, sizeof(cl_mem), &d_out_items);
	err |= clSetKernelArg(bfs_kernel, 3, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(bfs_kernel, 4, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(bfs_kernel, 5, sizeof(cl_mem), &d_dist);
	err |= clSetKernelArg(bfs_kernel, 6, sizeof(cl_mem), &d_in_frontier);
	err |= clSetKernelArg(bfs_kernel, 7, sizeof(cl_mem), &d_out_frontier);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
/*
	err = clSetKernelArg(exchange, 0, sizeof(cl_mem), &d_nitems);
	err |= clSetKernelArg(exchange, 1, sizeof(cl_mem), &d_out_items);
	err |= clSetKernelArg(exchange, 2, sizeof(cl_mem), &d_in_frontier);
	err |= clSetKernelArg(exchange, 3, sizeof(cl_mem), &d_out_frontier);
	err |= clSetKernelArg(exchange, 4, sizeof(cl_mem), &d_tmp);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
*/	

	int iter=0;

	Timer t;
	t.Start();

	err = clEnqueueNDRangeKernel(queue, insert, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

	err = clEnqueueReadBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 1, err code: %d\n", err); exit(1);}
	
		printf("zyy-nitems %d \n", nitems);
		//printf("zyy-column_indices %d, %d, %d, %d \n", h_column_indices[0], h_column_indices[1], h_column_indices[2],h_column_indices[3]);

	do {
		++iter;
	
		
		err = clEnqueueReadBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 1, err code: %d\n", err); exit(1);}
	
	//	int real = (nitems > out_items)? nitems : out_items;		
		if(nitems < 64)globalSize = 64;
		else if(nitems < 128)globalSize = 128;
		else if(nitems < 256)globalSize = 256;
		else if(nitems < 512)globalSize = 512;
		else if(nitems < 1024)globalSize = 1024;	
		else if(nitems < 2048)globalSize = 2048;	
		else if(nitems < 4096)globalSize = 4096;	
		else if(nitems < 8192)globalSize = 8192;	
		else if(nitems < 16384)globalSize = 16384;	
		else globalSize = ceil(m/(float)localSize)*localSize;
	
		err = clEnqueueNDRangeKernel(queue, bfs_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	
	//clFinish(queue);
	//t.Start();
		
		//err = clEnqueueReadBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
		//if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 1, err code: %d\n", err); exit(1);}
		
		err = clEnqueueReadBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}
		
	//unnecessary to use this, since we use copy buffer	
	/*	int real = (nitems > out_items)? nitems : out_items;		
		if(real < 64)globalSize = 64;
		else if(real < 128)globalSize = 128;
		else if(real < 256)globalSize = 256;
		else if(real < 512)globalSize = 512;
		else if(real < 1024)globalSize = 1024;	
		else if(real < 2048)globalSize = 2048;	
		else if(real < 4096)globalSize = 4096;	
		else if(real < 8192)globalSize = 8192;	
		else if(real < 16384)globalSize = 16384;	
		else globalSize = ceil(m/(float)localSize)*localSize;
	*/	

//		err = clEnqueueNDRangeKernel(queue, exchange, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
//		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}


		err = clEnqueueCopyBuffer(queue, d_in_frontier, d_tmp, 0, 0, nitems*sizeof(int), 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue copy buffer, err code: %d\n", err); exit(1);}

		err = clEnqueueCopyBuffer(queue, d_out_frontier, d_in_frontier, 0, 0, out_items*sizeof(int), 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue copy buffer, err code: %d\n", err); exit(1);}

		err = clEnqueueCopyBuffer(queue, d_tmp, d_out_frontier, 0, 0, nitems*sizeof(int), 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue copy buffer, err code: %d\n", err); exit(1);}


	//clFinish(queue);
	//t.Stop();
		err = clEnqueueReadBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}
		
		printf("out_items %d \n", out_items);
		
//		err = clEnqueueReadBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
//		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 4, err code: %d\n", err); exit(1);}
		
//		err = clEnqueueReadBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
//		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 5, err code: %d\n", err); exit(1);}
		
//		int *tmp = in_frontier;
//		in_frontier = out_frontier;
//		out_frontier = tmp;


		
//		err = clEnqueueWriteBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
//		if(err < 0){fprintf(stderr, "ERROR write buffer 3, err code: %d\n", err); exit(1);}
//		err = clEnqueueWriteBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
//		if(err < 0){fprintf(stderr, "ERROR write buffer 4, err code: %d\n", err); exit(1);}
		
		//d_tmp = d_in_frontier;
		//d_in_frontier = d_out_frontier;
		//d_out_frontier = d_tmp;		
		
		//d_nitems = d_out_items;
		
		
		err = clEnqueueWriteBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 5, err code: %d\n", err); exit(1);}
		
		

		err = clEnqueueWriteBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 6, err code: %d\n", err); exit(1);}


	//	printf("iteration end ...\n");
	} while(out_items > 0);

	clFinish(queue);
	
	t.Stop();
	
	printf("iter: %d \n", iter);
	
	err = clEnqueueReadBuffer(queue, d_dist, CL_TRUE, 0, sizeof(DistT) * m, h_dist , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}
	
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_dist);
	clReleaseMemObject(d_nitems);
	clReleaseMemObject(d_out_items);
	clReleaseMemObject(d_in_frontier);
	clReleaseMemObject(d_out_frontier);
	clReleaseMemObject(d_tmp);
	free(in_frontier);
	free(out_frontier);
	clReleaseProgram(program);
	clReleaseKernel(insert);
	clReleaseKernel(bfs_kernel);
	//clReleaseKernel(exchange);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());

	return;
}

