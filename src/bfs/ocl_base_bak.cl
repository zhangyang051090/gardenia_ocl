// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <vector>
#include <string.h>
#include "common.h"
#include "bfs.h"
//#include "worklisto.h"
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

void BFSSolver(int m, int nnz, int source, int *in_row_offsets, int *in_column_indices, int *h_row_offsets, int *h_column_indices, int *h_degree, DistT *h_dist) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(m/(float)localSize)*localSize;
	
	//Timer t;
	//t.Start();

	char *filechar = "/home/zy/gardinia_code/src/bfs/base.cl";
	int sourcesize = 1024*1024;
	char * source_1 = (char *)calloc(sourcesize, sizeof(char));
	FILE * fp = fopen(filechar, "rb");
	fread(source_1 + strlen(source_1), sourcesize, 1, fp);
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

	const char * slist[2] = {source_1, 0};
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
	
	bfs_kernel = clCreateKernel(program, "bfs_kernel", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	insert = clCreateKernel(program, "insert", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	int *in_frontier = (int *) malloc(m * sizeof(int));
	int *out_frontier = (int *) malloc(m * sizeof(int));
	
	int out_items = 0, nitems = 0;

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_dist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(DistT) * m, NULL, NULL);
	cl_mem d_nitems = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_in_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_out_frontier = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	//int d_in_items = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
	cl_mem d_out_items = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

	DistT zero = 0;
	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), h_row_offsets , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, h_column_indices, 0, NULL, NULL);
	//DistT * d_dist;
	err |= clEnqueueWriteBuffer(queue, d_dist, CL_TRUE, 0, sizeof(DistT) * m, h_dist, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_dist, CL_TRUE, 0, sizeof(zero), &zero, 0, NULL, NULL);//zy: with some doubts here.
	err |= clEnqueueWriteBuffer(queue, d_in_frontier, CL_TRUE, 0, sizeof(int) * m, in_frontier, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_out_frontier, CL_TRUE, 0, sizeof(int) * m, out_frontier, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	//Worklist2 queue1(m), queue2(m);
	//Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
	//vector<int> in_frontier(m);
	//vector<int> out_frontier(m);
	//int *in_frontier, *out_frontier;
	//int *in_frontier = (int *) malloc(m * sizeof(int));
	//int *out_frontier = (int *) malloc(m * sizeof(int));
	
	//int in_items = m, out_items = 0, nitems = 1;


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
        //clFinish(queue);

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-bef is %f\n", y[i]);


	int iter=0;
	//int nitems = 1;

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
	
		printf("iteration start....\n");
		//err = clEnqueueWriteBuffer(queue, d_diff, CL_TRUE, 0, sizeof(float), &diff, 0, NULL, NULL);
		//if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}
	
	//t.Start();
		err = clEnqueueNDRangeKernel(queue, bfs_kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
		//nitems = out_frontier->nitems();
			
		err = clEnqueueReadBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}
		
		printf("zyy-out_items %d \n", out_items);
		//nitems = index;
		
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
		//if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}
		//out_frontier.clear();	
		
		//err = clEnqueueReadBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &nitems, 0, NULL, NULL);
		//if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 2, err code: %d\n", err); exit(1);}
		
		//int *tmp_num = &d_nitems;
		
		err = clEnqueueWriteBuffer(queue, d_nitems, CL_TRUE, 0, sizeof(int), &out_items, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 5, err code: %d\n", err); exit(1);}
		
		//err = clEnqueueWriteBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
		//if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}
		//&d_nitems = &d_out_items;
		//&d_out_items = tmp_num;
		

		err = clEnqueueWriteBuffer(queue, d_out_items, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer 6, err code: %d\n", err); exit(1);}


	//clFinish(queue);
	//t.Stop();
	//printf("\tzyy-runtime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
		printf("end of one iteration ...\n");
	} while(out_items > 0);
	//t.Stop();

	clFinish(queue);
	
	t.Stop();
	
	err = clEnqueueReadBuffer(queue, d_dist, CL_TRUE, 0, sizeof(DistT) * m, h_dist , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer 3, err code: %d\n", err); exit(1);}

	

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-after is %f\n", y[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_dist);
	clReleaseMemObject(d_nitems);
	clReleaseMemObject(d_out_items);
	clReleaseMemObject(d_in_frontier);
	clReleaseMemObject(d_out_frontier);
	free(in_frontier);
	free(out_frontier);
	//clReleaseMemObject(d_degree);
	//clReleaseMemObject(d_scores);
	//clReleaseMemObject(d_diff);
	clReleaseProgram(program);
	//clReleaseProgram(program_gather);
	clReleaseKernel(insert);
	clReleaseKernel(bfs_kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, t.Millisecs());

	return;
}

