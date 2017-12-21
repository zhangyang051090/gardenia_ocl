// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <vector>
#include <string.h>
#include "common.h"
#include <timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#define SYMGS_VARIANT "ocl_base"

cl_platform_id cpPlatform[32];
cl_uint num_platforms;
cl_device_id device_id[5];
cl_uint num_devices;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;

const char *kernelSource =																"\n" \
"__kernel void symgs_kernel(int m, int n, __global int * Ap, __global int * Aj, __global int * indices, __global float * Ax, __global float * x, __global float * b, __global int * color_offsets, int size) \n" \
"{																			\n" \
"																			\n" \
"	int i = get_global_id(0);															\n" \
"																			\n" \
"	//for(size_t j = 0; j < size-1; j++){														\n" \
"	if( i < size ){																	\n" \
"		//for ( int i = color_offsets[j]; i < color_offsets[j+1]; i += 1){									\n" \
"			int inew = indices[i+n];														\n" \
"			int row_begin = Ap[inew];													\n" \
"			int row_end = Ap[inew + 1];													\n" \
"			float rsum = 0;															\n" \
"			float diag = 0;															\n" \
"			for (int jj = row_begin; jj < row_end; jj ++){											\n" \
"				const int j = Aj[jj];													\n" \
"				if (inew == j) diag = Ax[jj];												\n" \
"				else rsum += x[j] * Ax[jj];												\n" \
"			}																\n" \
"																			\n" \
"		if(diag != 0) x[inew] = (b[inew] - rsum) / diag;											\n" \
"	  																		\n" \
"	}																		\n" \
"	/*																		\n" \
"	//for(size_t j = size-1; j > 0; j--){														\n" \
"	if( j < size-1 ){																\n" \
"		for ( int i = color_offsets[j-1]; i < color_offsets[j]; i += 1){									\n" \
"			int inew = indices[i];														\n" \
"			int row_begin = Ap[inew];													\n" \
"			int row_end = Ap[inew + 1];													\n" \
"			float rsum = 0;															\n" \
"			float diag = 0;															\n" \
"			for (int jj = row_begin; jj < row_end; jj ++){											\n" \
"				const int j = Aj[jj];													\n" \
"				if (inew == j) diag = Ax[jj];												\n" \
"				else rsum += x[j] * Ax[jj];												\n" \
"			}																\n" \
"																			\n" \
"		if(diag != 0) x[inew] = (b[inew] - rsum) / diag;											\n" \
"	  	}																	\n" \
"	}*/																		\n" \
"	return;																		\n" \
"}																			\n" \
																			"\n" ;

void SymGSSolver(int m, int nnz, int *Ap, int *Aj, int *indices, ValueType *Ax, ValueType *x, ValueType *b, std::vector<int> color_offsets) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(m/(float)localSize)*localSize;
	//int num_colors;
	//Timer t;
	//t.Start();

	//char *filechar = "/home/zy/gardinia_code/src/spmv/base.cl";
	//int sourcesize = 1024*1024;
	//char * source = (char *)calloc(sourcesize, sizeof(char));
	//FILE * fp = fopen(filechar, "rb");
	//fread(source + strlen(source), sourcesize, 1, fp);
	//fclose(fp);
	
err = clGetPlatformIDs(0, NULL, &num_platforms);
	printf("\nNumber of platforms:\t%u\n\n", num_platforms);	
	
	err = clGetPlatformIDs(num_platforms, cpPlatform, NULL);
	printf("\nplatform id:\t%u\n\n", cpPlatform[0]);	
	printf("\nplatform id:\t%u\n\n", cpPlatform[1]);	
	
	size_t size_name;
	size_t maxComputeUnits;
	err = clGetPlatformInfo(cpPlatform[0], CL_PLATFORM_NAME, 0, NULL, &size_name);
	char * name_p = (char *)alloca(sizeof(char) * size_name);
	//char * info = (char *)alloca(sizeof(char) * size);
	err = clGetPlatformInfo(cpPlatform[0], CL_PLATFORM_NAME, size_name, name_p, NULL);
	printf("\nplatform name:\t%s\n\n", name_p);	

	err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &num_devices);
	if(err < 0){fprintf(stderr, "ERROR get num of devices, err code: %d\n", err); exit(1);}
	printf("number of devices are %d\n", num_devices);
	
	
	err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id[0], NULL);
	if(err < 0){fprintf(stderr, "ERROR get devices id, err code: %d\n", err); exit(1);}
	printf("device id is %d\n", device_id[0]);

	err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id[1], NULL);
	if(err < 0){fprintf(stderr, "ERROR get devices id, err code: %d\n", err); exit(1);}
	printf("device id is %d\n", device_id[1]);

	err = clGetDeviceInfo(device_id[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, &size_name);
	if(err < 0){fprintf(stderr, "ERROR get devices info, err code: %d\n", err); exit(1);}
	printf("max compute units is %d\n", maxComputeUnits);

	
	err = clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, 0, NULL, &size_name);
	char * name = (char *)alloca(sizeof(char) * size_name);
	//char * info = (char *)alloca(sizeof(char) * size);
	err = clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, size_name*sizeof(char), name, NULL);
	printf("\ndev name:\t%s\n\n", name);	

	//err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(char), &name, &size);
	//if(err < 0){fprintf(stderr, "ERROR get devices info, err code: %d\n", err); exit(1);}
	//printf("device name is %c\n", name_1[0]);
	
	context = clCreateContext(0, 1, &device_id[0], NULL, NULL, &err);
	queue = clCreateCommandQueue(context, device_id[0], 0, &err);
	if(err < 0){fprintf(stderr, "ERROR command queue, err code: %d\n", err); exit(1);}

//	const char * slist[2] = {source_1, 0};
	program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
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
	

	kernel = clCreateKernel(program, "symgs_kernel", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	cl_mem d_Ap = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_Aj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_indices = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_Ax = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ValueType) * nnz, NULL, NULL);
	cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ValueType) * m, NULL, NULL);
	cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ValueType) * m, NULL, NULL);
	cl_mem d_color_offsets = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * color_offsets.size(), NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_Ap, CL_TRUE, 0, sizeof(int) * (m+1), Ap , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_Aj, CL_TRUE, 0, sizeof(int) * nnz, Aj, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_indices, CL_TRUE, 0, sizeof(int) * m, indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_Ax, CL_TRUE, 0, sizeof(ValueType) * nnz, Ax, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, sizeof(ValueType) * m, x, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, sizeof(ValueType) * m, b, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_color_offsets, CL_TRUE, 0, sizeof(int)*color_offsets.size() , color_offsets.data(), 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	//int size = color_offsets.size();
	int size;
	int n;
	err = clSetKernelArg(kernel, 0, sizeof(int), &m);
	//err |= clSetKernelArg(kernel, 1, sizeof(int), &n);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_Ap);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_Aj);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_indices);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_Ax);
	err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_x);
	err |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_b);
	err |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_color_offsets);
	//err |= clSetKernelArg(kernel, 9, sizeof(int), &size);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	
        //clFinish(queue);

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-bef is %f\n", y[i]);
	
	Timer t;
	t.Start();
	
	for(int i = 0; i < color_offsets.size()-1; i++){
		size = color_offsets[i+1] - color_offsets[i];
		err = clSetKernelArg(kernel, 9, sizeof(int), &size);
		if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	
		n = color_offsets[i];

		err = clSetKernelArg(kernel, 1, sizeof(int), &n);
		if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	
		globalSize = ceil(size/(float)localSize)*localSize;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	}
	
	for(int i = color_offsets.size()-1; i > 0; i--){
		size = color_offsets[i] - color_offsets[i-1];
		err = clSetKernelArg(kernel, 9, sizeof(int), &size);
		if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
			
		n = color_offsets[i-1];

		err = clSetKernelArg(kernel, 1, sizeof(int), &n);
		if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
		
		globalSize = ceil(size/(float)localSize)*localSize;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	}

	clFinish(queue);
	
	t.Stop();
	
	err = clEnqueueReadBuffer(queue, d_x, CL_TRUE, 0, sizeof(ValueType) * m, x , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}

	
	//for(int i = 0; i < 50; i++)
	//	printf("value of y-after is %f\n", y[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_Ap);
	clReleaseMemObject(d_Aj);
	clReleaseMemObject(d_indices);
	clReleaseMemObject(d_Ax);
	clReleaseMemObject(d_x);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_color_offsets);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
	printf("\truntime [%s] = %f ms.\n", SYMGS_VARIANT, t.Millisecs());

	return;
}

