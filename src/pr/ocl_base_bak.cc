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
cl_program program_calc_contrib;
cl_program program_gather;
cl_kernel calc_contrib;
cl_kernel gather;
//#pragma OPENCL EXTENSION cl_khr_global_int64_base_atomics: enable	
/*
"	inline void AtomicAdd(volatile __global float * diff, const float error){  			
"		union{																	
"			unsigned int intVal;														
"			float floatVal;															
"		}newVal;																
"																			
"		union{																	
"			unsigned int intVal;														
"			float floatVal;															
"		}prevVal;																
"			do{																
"				prevVal.floatVal = *diff;												
"				newVal.floatVal =prevVal.floatVal + error;										
"			}while(atomic_cmpxchg((volatile __global unsigned int *)diff,									
"						prevVal.intVal, newVal.intVal)!=prevVal.intVal);							
"	}																	
"

const char *kernelSource_calc_contrib =																"\n" \
"__kernel void calc_contrib(int m, int nnz, __global int * row_offsets, __global int * column_indices, __global float * outgoing_contrib, __global int * degree, __global float * scores, __global float * diff) \n" \
"{																			\n" \
"																			\n" \
"	int src = get_global_id(0);															\n" \
"																			\n" \
"		if( src < m )																\n" \
"			outgoing_contrib[src] = scores[src] / degree[src];										\n" \
"}																			\n" \
																			"\n" ;



const char *kernelSource_gather =																"\n" \
"__kernel void gather(int m, int nnz, __global int * row_offsets, __global int * column_indices, __global float * outgoing_contrib, __global int * degree, __global float * scores, __global float * diff) \n" \
"{																			\n" \
"																			\n" \
"		union{																	\n" \
"			unsigned int intVal;														\n" \
"			float floatVal;															\n" \
"		}newVal;																\n" \
"																			\n" \
"		union{																	\n" \
"			unsigned int intVal;														\n" \
"			float floatVal;															\n" \
"		}prevVal;																\n" \
"																			\n" \
"	int src = get_global_id(0);															\n" \
"																			\n" \
"	const float base_score = (1.0f - 0.85) / m;													\n" \
"																			\n" \
"		float error = 0.0;															\n" \
"		if( src < m ){																\n" \
"			float incoming_total = 0;													\n" \
"			int row_begin = row_offsets[src];												\n" \
"			int row_end = row_offsets[src + 1];												\n" \
"			for (int offset = row_begin; offset < row_end; offset ++){									\n" \
"				int dst = column_indices[offset];											\n" \
"				incoming_total += outgoing_contrib[dst];										\n" \
"			}																\n" \
"			float old_score = scores[src];													\n" \
"			scores[src] = base_score + 0.85 * incoming_total;										\n" \
"			error += fabs(scores[src] - old_score);												\n" \
"		//	Atomic_Add(diff, error);												\n" \
"			//*diff = error;														\n" \
"																			\n" \
"			do{																\n" \
"				prevVal.floatVal = *diff;												\n" \
"				newVal.floatVal =prevVal.floatVal + error;										\n" \
"			}while(atomic_cmpxchg((volatile __global unsigned int *)diff,									\n" \
"						prevVal.intVal, newVal.intVal)!=prevVal.intVal);							\n" \
"		}																	\n" \
"																			\n" \
"	return;																		\n" \
"}																			\n" \
																			"\n" ;
*/
void PRSolver(int m, int nnz, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, int *degree, ScoreT *h_scores) {
	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(m/(float)localSize)*localSize;
	
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


	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_outgoing_contrib = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * m, NULL, NULL);
	cl_mem d_degree = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem d_scores = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT) * m, NULL, NULL);
	cl_mem d_diff = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), in_row_offsets , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, in_column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_scores, CL_TRUE, 0, sizeof(ScoreT) * m, h_scores, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_degree, CL_TRUE, 0, sizeof(int) * m, degree, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	err = clSetKernelArg(calc_contrib, 0, sizeof(int), &m);
	err |= clSetKernelArg(calc_contrib, 1, sizeof(int), &nnz);
	err |= clSetKernelArg(calc_contrib, 2, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(calc_contrib, 3, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(calc_contrib, 4, sizeof(cl_mem), &d_outgoing_contrib);
	err |= clSetKernelArg(calc_contrib, 5, sizeof(cl_mem), &d_degree);
	err |= clSetKernelArg(calc_contrib, 6, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(calc_contrib, 7, sizeof(cl_mem), &d_diff);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}


		
	err = clSetKernelArg(gather, 0, sizeof(int), &m);
	err |= clSetKernelArg(gather, 1, sizeof(int), &nnz);
	err |= clSetKernelArg(gather, 2, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(gather, 3, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(gather, 4, sizeof(cl_mem), &d_outgoing_contrib);
	err |= clSetKernelArg(gather, 5, sizeof(cl_mem), &d_degree);
	err |= clSetKernelArg(gather, 6, sizeof(cl_mem), &d_scores);
	err |= clSetKernelArg(gather, 7, sizeof(cl_mem), &d_diff);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
        //clFinish(queue);

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-bef is %f\n", y[i]);


	int iter=0;
	float diff;	
	Timer t;
	t.Start();

	do {
		++iter;
		diff = 0;
	
		
		err = clEnqueueNDRangeKernel(queue, calc_contrib, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
	
		err = clEnqueueWriteBuffer(queue, d_diff, CL_TRUE, 0, sizeof(float), &diff, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}
	
		err = clEnqueueNDRangeKernel(queue, gather, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

	
		err = clEnqueueReadBuffer(queue, d_diff, CL_TRUE, 0, sizeof(float), &diff , 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}

		printf(" %2d   %lf\n", iter, diff);
	} while(diff > EPSILON && iter < MAX_ITER);
	//t.Stop();

	clFinish(queue);
	
	t.Stop();
	
	err = clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, sizeof(ScoreT) * m, h_scores , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}

	

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-after is %f\n", y[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_outgoing_contrib);
	clReleaseMemObject(d_degree);
	clReleaseMemObject(d_scores);
	clReleaseMemObject(d_diff);
	clReleaseProgram(program);
	//clReleaseProgram(program_gather);
	clReleaseKernel(calc_contrib);
	clReleaseKernel(gather);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());

	return;
}

