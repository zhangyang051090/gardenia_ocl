// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "cc.h"
#include <unordered_map>
#include <stack>
#include <vector>
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
cl_program program_1;
cl_program program_2;
cl_kernel kernel_1;
cl_kernel kernel_2;

const char *kernelSource_1 =																"\n" \
"__kernel void cc_scatter_kernel(int m, int nnz, __global int * row_offsets, __global int * column_indices,  __global int * comp, __global bool * changed)        		\n" \
"{																			\n" \
"	//bool *changed = true;																\n" \
"	//int iter = 0;																	\n" \
"																			\n" \
"	   int src = get_global_id(0);															\n" \
"	//while (change){																	\n" \
"	//	change = false;																\n" \
"	//	iter++;																	\n" \
"		if(src < m){																\n" \
"			int comp_src = comp[src];													\n" \
"			int row_begin = row_offsets[src];												\n" \
"			int row_end = row_offsets[src + 1];												\n" \
"			for (int offset = row_begin; offset < row_end; offset ++){									\n" \
"				int dst = column_indices[offset];											\n" \
"				int comp_dst = comp[dst];												\n" \
"				if((comp_src < comp_dst) && (comp_dst == comp[comp_dst])){								\n" \
"					*changed = true;													\n" \
"					comp[comp_dst] = comp_src;			  								\n" \
"				}															\n" \
"			}																\n" \
"	         }																	\n" \
"																			\n" \
"}																			\n" \
																			"\n" ;

const char *kernelSource_2 =																"\n" \
"__kernel void cc_update_kernel(int m, int nnz, __global int * row_offsets, __global int * column_indices,  __global int * comp)        		\n" \
"{																			\n" \
"	   int src = get_global_id(0);															\n" \
"		if(src < m){																\n" \
"			while(comp[src] != comp[comp[src]]){												\n" \
"				comp[src] = comp[comp[src]];												\n" \
"			}																\n" \
"	        }																	\n" \
"}																			\n" \
																			"\n" ;

void CCSolver(int m, int nnz, int *row_offsets, int *column_indices, CompT *comp) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(m/(float)localSize)*localSize;
	
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
	program_1 = clCreateProgramWithSource(context, 1, (const char **)& kernelSource_1, NULL, &err);
	if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}
	
	program_2 = clCreateProgramWithSource(context, 1, (const char **)& kernelSource_2, NULL, &err);
	if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}
		
	err = clBuildProgram(program_1, 0, NULL, NULL, NULL, NULL);
  err = clBuildProgram(program_2, 0, NULL, NULL, NULL, NULL);
  
	if(err < 0){
		size_t len;
		char buffer[1000];
	
		printf("error: build program failure\n");
		clGetProgramBuildInfo(program_1, device_id[0], CL_PROGRAM_BUILD_LOG,1000, buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}
	


	kernel_1 = clCreateKernel(program_1, "cc_scatter_kernel", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	kernel_2 = clCreateKernel(program_2, "cc_update_kernel", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_comp = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(CompT) * m, NULL, NULL);
	cl_mem d_changed = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool), NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), row_offsets , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_comp, CL_TRUE, 0, sizeof(CompT) * m, comp, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	err = clSetKernelArg(kernel_1, 0, sizeof(int), &m);
	err |= clSetKernelArg(kernel_1, 1, sizeof(int), &nnz);
	err |= clSetKernelArg(kernel_1, 2, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(kernel_1, 3, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(kernel_1, 4, sizeof(cl_mem), &d_comp);
	err |= clSetKernelArg(kernel_1, 5, sizeof(cl_mem), &d_changed);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	
	err = clSetKernelArg(kernel_2, 0, sizeof(int), &m);
	err |= clSetKernelArg(kernel_2, 1, sizeof(int), &nnz);
	err |= clSetKernelArg(kernel_2, 2, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(kernel_2, 3, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(kernel_2, 4, sizeof(cl_mem), &d_comp);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
        //clFinish(queue);

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-bef is %f\n", y[i]);
	
	Timer t;
	t.Start();

	int iter = 0;
	bool changed;

	do {
		++ iter;
		changed = false;
		err = clEnqueueWriteBuffer(queue, d_changed, CL_TRUE, 0, sizeof(changed), &changed, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}
	

		err = clEnqueueNDRangeKernel(queue, kernel_1, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

		err = clEnqueueNDRangeKernel(queue, kernel_2, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	
		err = clEnqueueReadBuffer(queue, d_changed, CL_TRUE, 0, sizeof(changed), &changed, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}
	} while (changed);
	


	//t.Stop();

	clFinish(queue);
	
	t.Stop();
	
	err = clEnqueueReadBuffer(queue, d_comp, CL_TRUE, 0, sizeof(CompT) * m, comp, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}

	
	//for(int i = 0; i < 50; i++)
	//	printf("value of y-after is %f\n", y[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_comp);
	clReleaseMemObject(d_changed);
	clReleaseProgram(program_1);
	clReleaseProgram(program_2);
	clReleaseKernel(kernel_1);
	clReleaseKernel(kernel_2);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());

	return;
}

