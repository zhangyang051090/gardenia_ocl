// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <vector>
#include <string.h>
#include "common.h"
#include "bc.h"
#include <timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#define BC_VARIANT "ocl_base"

cl_platform_id cpPlatform[32];
cl_uint num_platforms;
cl_device_id device_id;
cl_uint num_devices;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel bc_forward;
cl_kernel bc_reverse;
cl_kernel initialize;
cl_kernel bc_normalize;
cl_kernel insert;
cl_kernel max_element;
cl_kernel push_frontier;

void BCSolver(int m, int nnz, int source, int *h_row_offsets, int *h_column_indices, ScoreT *h_scores) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(m/(float)localSize)*localSize;
	
	char *filechar = "/home/zy/gardinia_code/src/bc/base.cl";
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
	
	initialize = clCreateKernel(program, "initialize", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}
	
	bc_forward = clCreateKernel(program, "bc_forward", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	bc_reverse = clCreateKernel(program, "bc_reverse", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}
	
	insert = clCreateKernel(program, "insert", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	push_frontier = clCreateKernel(program, "push_frontier", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}
	
	bc_normalize = clCreateKernel(program, "bc_normalize", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}
	
	max_element = clCreateKernel(program, "max_element", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}
	
	ScoreT *h_deltas = (ScoreT *) malloc(m * sizeof(ScoreT));
	int *h_path_counts = (int *) malloc(m * sizeof(int));
	
	int *in_frontier = (int *) malloc(m * sizeof(int));
	int *out_frontier = (int *) malloc(m * sizeof(int));
	

	for(int i = 0; i < m; i++) h_deltas[i] = 0;
	for(int i = 0; i < m; i++) h_path_counts[i] = 0;

	int depth = 0;
	int out_items = 0, nitems = 0;
	int frontiers_len = 0;
	int zero = 0;
	int start;
	ScoreT max_score;
	vector<int> depth_index;
	depth_index.push_back(0);

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT) * m, NULL, NULL);
	cl_mem d_deltas = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT) * m, NULL, NULL);
	cl_mem d_nitems = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_path_counts = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_depths = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_max_score = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT), NULL, NULL);
	cl_mem d_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (m + 1), NULL, NULL);
	cl_mem d_in_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_out_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_out_items = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	//DistT zero = 0;
	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), h_row_offsets , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, h_column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_scores, CL_TRUE, 0, sizeof(ScoreT) * m, h_scores, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_deltas, CL_TRUE, 0, sizeof(ScoreT) * m, h_deltas, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_path_counts, CL_TRUE, 0, sizeof(int) * m, h_path_counts, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}


	err = clSetKernelArg(initialize, 0, sizeof(int), &m);
	err |= clSetKernelArg(initialize, 1, sizeof(cl_mem), &d_depths);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	
	err = clSetKernelArg(insert, 0, sizeof(int), &source);
	err |= clSetKernelArg(insert, 1, sizeof(cl_mem), &d_nitems);
	err |= clSetKernelArg(insert, 2, sizeof(cl_mem), &d_path_counts);
	err |= clSetKernelArg(insert, 3, sizeof(cl_mem), &d_depths);
	err |= clSetKernelArg(insert, 4, sizeof(cl_mem), &d_in_frontier);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}


		
	err = clSetKernelArg(bc_forward, 0, sizeof(int), &m);
	err |= clSetKernelArg(bc_forward, 1, sizeof(cl_mem), &d_nitems);
	err |= clSetKernelArg(bc_forward, 2, sizeof(cl_mem), &d_out_items);
	err |= clSetKernelArg(bc_forward, 3, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(bc_forward, 4, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(bc_forward, 5, sizeof(cl_mem), &d_path_counts);
	err |= clSetKernelArg(bc_forward, 6, sizeof(cl_mem), &d_depths);
	err |= clSetKernelArg(bc_forward, 7, sizeof(int), &depth);
	err |= clSetKernelArg(bc_forward, 8, sizeof(cl_mem), &d_in_frontier);
	err |= clSetKernelArg(bc_forward, 9, sizeof(cl_mem), &d_out_frontier);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}

	err = clSetKernelArg(bc_reverse, 0, sizeof(cl_mem), &d_nitems);
	err |= clSetKernelArg(bc_reverse, 1, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(bc_reverse, 2, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(bc_reverse, 3, sizeof(int), &start);
	err |= clSetKernelArg(bc_reverse, 4, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(bc_reverse, 5, sizeof(cl_mem), &d_path_counts);
	err |= clSetKernelArg(bc_reverse, 6, sizeof(cl_mem), &d_depths);
	err |= clSetKernelArg(bc_reverse, 7, sizeof(int), &depth);
	err |= clSetKernelArg(bc_reverse, 8, sizeof(cl_mem), &d_deltas);
	err |= clSetKernelArg(bc_reverse, 9, sizeof(cl_mem), &d_in_frontier);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}

	
	err = clSetKernelArg(bc_normalize, 0, sizeof(int), &m);
	err |= clSetKernelArg(bc_normalize, 1, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(bc_normalize, 2, sizeof(cl_mem), &d_max_score);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}

	err = clSetKernelArg(push_frontier, 0, sizeof(cl_mem), &d_nitems);
	err |= clSetKernelArg(push_frontier, 1, sizeof(cl_mem), &d_in_frontier);
	err |= clSetKernelArg(push_frontier, 2, sizeof(cl_mem), &d_frontier);
	err |= clSetKernelArg(push_frontier, 3, sizeof(int), &frontiers_len);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	
	err = clSetKernelArg(max_element, 0, sizeof(int), &m);
	err |= clSetKernelArg(max_element, 1, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(max_element, 2, sizeof(cl_mem), &d_max_score);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	
	int iter=0;

	err = clEnqueueNDRangeKernel(queue, initialize, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

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
	
		printf("iteration start....\n");
		
		
		//printf("zyy-out_items %d \n", out_items);
	
		err = clEnqueueNDRangeKernel(queue, push_frontier, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	
		printf("push finished....\n");
		
		frontiers_len += nitems;
		depth_index.push_back(frontiers_len);
		depth++;
	
		printf("bc_forward begins....\n");
		err = clEnqueueNDRangeKernel(queue, bc_forward, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
		
		//zy	
		err = clEnqueueReadBuffer(queue, d_path_counts, CL_TRUE, 0, sizeof(int) * m, h_path_counts , 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 4, err code: %d\n", err); exit(1);}
	for(int i = 0; i < 2; i++)
		printf(" %d ", h_path_counts[i]);
		
		err = clEnqueueReadBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}
		
		printf("zyy-out_items %d \n", out_items);
		
		err = clEnqueueReadBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 4, err code: %d\n", err); exit(1);}
		
		err = clEnqueueReadBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 5, err code: %d\n", err); exit(1);}
		
		int *tmp = in_frontier;
		in_frontier = out_frontier;
		out_frontier = tmp;


		
		err = clEnqueueWriteBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 3, err code: %d\n", err); exit(1);}
		err = clEnqueueWriteBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 4, err code: %d\n", err); exit(1);}
		
		
		
		err = clEnqueueWriteBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 5, err code: %d\n", err); exit(1);}
		
		

		err = clEnqueueWriteBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 6, err code: %d\n", err); exit(1);}


		printf("iteration end ...\n");
	} while(out_items > 0);

	printf("iter %d\n", iter);	

	printf("depth_index_size %d\n", depth_index.size());

		err = clEnqueueReadBuffer(queue, d_path_counts, CL_TRUE, 0, sizeof(int) * m, h_path_counts , 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 4, err code: %d\n", err); exit(1);}
	//for(int i = 0; i < m; i++)
	//	printf(" %d ", h_path_counts[i]);

	for (int d = depth_index.size() - 2; d >= 0; d--){
		nitems = depth_index[d+1] - depth_index[d];
	
		start = depth_index[d];
	
		err = clEnqueueNDRangeKernel(queue, bc_reverse, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

	}

	err = clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, sizeof(ScoreT) * m, h_scores , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}
	//for(int i = 0; i < m; i++)
	//	printf(" %f ", h_scores[i]);
		

		//printf("bc_reverse finished ...\n");
	//	ScoreT *d_max_score;

		err = clEnqueueNDRangeKernel(queue, max_element, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

		//printf("max_element finished ...\n");

		err = clEnqueueReadBuffer(queue, d_max_score, CL_TRUE, 0, sizeof(ScoreT), &max_score, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}

		printf("max_score %f\n", max_score);

		err = clEnqueueNDRangeKernel(queue, bc_normalize, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

		//printf("bc_normalize finished ...\n");
	
	clFinish(queue);
	
	t.Stop();
	
	err = clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, sizeof(ScoreT) * m, h_scores , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}
	
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_path_counts);
	clReleaseMemObject(d_depths);
	clReleaseMemObject(d_deltas);
	clReleaseMemObject(d_max_score);
	clReleaseMemObject(d_frontier);
	clReleaseMemObject(d_nitems);
	clReleaseMemObject(d_out_items);
	clReleaseMemObject(d_in_frontier);
	clReleaseMemObject(d_out_frontier);
	free(in_frontier);
	free(out_frontier);
	free(h_deltas);
	free(h_path_counts);
	clReleaseProgram(program);
	clReleaseKernel(initialize);
	clReleaseKernel(insert);
	clReleaseKernel(bc_forward);
	clReleaseKernel(bc_reverse);
	clReleaseKernel(bc_normalize);
	clReleaseKernel(max_element);
	clReleaseKernel(push_frontier);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	
	printf("\truntime [%s] = %f ms.\n", BC_VARIANT, t.Millisecs());

	return;
}

