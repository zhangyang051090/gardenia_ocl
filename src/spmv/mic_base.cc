// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <string.h>
#include <timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#define SPMV_VARIANT "ocl_base"

cl_platform_id cpPlatform[32];
cl_uint num_platforms;
cl_device_id device_id[5];
cl_uint num_devices;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;

const char *kernelSource =																"\n" \
"__kernel void spmv_kernel(int num_rows, __global int * Ap, __global int * Aj, __global float * Ax, __global float * x, __global float * y)           	\n" \
"{																			\n" \
"	int idx = get_global_id(0);															\n" \
"	//int jj = get_global_id(1);															\n" \
"	if(idx < num_rows){																\n" \
"		int row_begin = Ap[idx];														\n" \
"		int row_end = Ap[idx+1];														\n" \
"		float sum = y[idx];															\n" \
"																			\n" \
"		for(int jj = row_begin; jj < row_end; jj++){												\n" \
"		//if(row_begin-1 < jj < row_end){													\n" \
"			int j = Aj[jj];															\n" \
"			sum += x[j] * Ax[jj];														\n" \
"		}																	\n" \
"		y[idx] = sum;																\n" \
"	}																		\n" \
"																			\n" \
"														  					\n" \
"}																			\n" \
																			"\n" ;

void SpmvSolver(int num_rows, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(num_rows/(float)localSize)*localSize;
	
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
	
	size_t size;
	size_t maxComputeUnits;
	err = clGetPlatformInfo(cpPlatform[0], CL_PLATFORM_NAME, 0, NULL, &size);
	char * name_p = (char *)alloca(sizeof(char) * size);
	//char * info = (char *)alloca(sizeof(char) * size);
	err = clGetPlatformInfo(cpPlatform[0], CL_PLATFORM_NAME, size, name_p, NULL);
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

	err = clGetDeviceInfo(device_id[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, &size);
	if(err < 0){fprintf(stderr, "ERROR get devices info, err code: %d\n", err); exit(1);}
	printf("max compute units is %d\n", maxComputeUnits);

	
	err = clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, 0, NULL, &size);
	char * name = (char *)alloca(sizeof(char) * size);
	//char * info = (char *)alloca(sizeof(char) * size);
	err = clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, size*sizeof(char), name, NULL);
	printf("\ndev name:\t%s\n\n", name);	

	//err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(char), &name, &size);
	//if(err < 0){fprintf(stderr, "ERROR get devices info, err code: %d\n", err); exit(1);}
	//printf("device name is %c\n", name_1[0]);
	
	context = clCreateContext(0, 1, &device_id[0], NULL, NULL, &err);
	queue = clCreateCommandQueue(context, device_id[0], 0, &err);
	if(err < 0){fprintf(stderr, "ERROR command queue, err code: %d\n", err); exit(1);}

	//const char * slist[2] = {source_1, 0};
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
	
	kernel = clCreateKernel(program, "spmv_kernel", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	cl_mem d_Ap = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (num_rows+1), NULL, NULL);
	cl_mem d_Aj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_Ax = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(ValueType) * nnz, NULL, NULL);
	cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(ValueType) * nnz, NULL, NULL);
	cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ValueType) * num_rows, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_Ap, CL_TRUE, 0, sizeof(int) * (num_rows+1), Ap, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_Aj, CL_TRUE, 0, sizeof(int) * nnz, Aj, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_Ax, CL_TRUE, 0, sizeof(ValueType) * nnz, Ax, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_x, CL_TRUE, 0, sizeof(ValueType) * nnz, x, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_y, CL_TRUE, 0, sizeof(ValueType) * num_rows, y, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	err = clSetKernelArg(kernel, 0, sizeof(int), &num_rows);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_Ap);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_Aj);
	err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_Ax);
	err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_x);
	err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_y);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-bef is %f\n", y[i]);
	
	Timer t;
	t.Start();

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

	//t.Stop();

	clFinish(queue);
	t.Stop();

	err = clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, sizeof(ValueType) * num_rows, y, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}

	
	//for(int i = 0; i < 50; i++)
	//	printf("value of y-after is %f\n", y[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_Ap);
	clReleaseMemObject(d_Aj);
	clReleaseMemObject(d_Ax);
	clReleaseMemObject(d_x);
	clReleaseMemObject(d_y);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());

	return;
}

