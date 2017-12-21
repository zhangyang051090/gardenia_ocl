// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <vector>
#include <string.h>
#include "common.h"
#include "sgd.h"
#include <timer.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#define SGD_VARIANT "ocl_base"

cl_platform_id cpPlatform[32];
cl_uint num_platforms;
cl_device_id device_id;
cl_uint num_devices;
cl_context context;
cl_command_queue queue;
cl_program program_cal_delta;
cl_program program_cal_rmse;
cl_kernel kernel_cal_delta;
cl_kernel kernel_cal_rmse;

const char *kernelSource_cal_delta =																"\n" \
"__kernel void sgd_kernel_cal_delta(int m, int n, __global int * row_offsets, __global int * column_indices, __global float * rating, __global float * user_lv, __global float * item_lv, float lambda, float step,  __global int * ordering)  				                                                                                              \n" \
"{																			\n" \
"		int id = get_global_id(0);																		\n" \
"																			\n" \
"		if( id < m ){														\n" \
"			int src = ordering[id];														\n" \
"			int row_begin = row_offsets[src];												\n" \
"			int row_end = row_offsets[src+1];												\n" \
"																			\n" \
"			for ( int offset = row_begin; offset < row_end; ++ offset){									\n" \
"				int dst = column_indices[offset];											\n" \
"				float estimate = 0;													\n" \
"				for (int i = 0; i < 128; i++){												\n" \
"					estimate += user_lv[src*128+i] * item_lv[dst*128+i];								\n" \
"				}															\n" \
"				float delta = rating[offset] - estimate;										\n" \
"				for (int i = 0; i < 128; i++){												\n" \
"					float p_s = user_lv[src*128+i];											\n" \
"					float p_d = item_lv[dst*128+i];											\n" \
"					user_lv[src*128+i] += step * (-lambda * p_s + p_d * delta);							\n" \
"					item_lv[dst*128+i] += step * (-lambda * p_d + p_s * delta);							\n" \
"				}															\n" \
"			}																\n" \
"		}																	\n" \
"																			\n" \
"}																			\n" \
																			"\n" ;

const char *kernelSource_cal_rmse =																"\n" \
"__kernel void sgd_kernel_cal_rmse(int m, int n, __global int * row_offsets, __global int * column_indices, __global float * rating, __global float * user_lv, __global float * item_lv, __global float *total_error)  				                                                                                              \n" \
"{																			\n" \
"																			\n" \
"	//int n = get_global_id(0);															\n" \
"	//int iter = 0;																	\n" \
"	float total = 0.0;																\n" \
"																			\n" \
"	int src = get_global_id(0);															\n" \
"		//for (int src = 0; src < m; src ++){													\n" \
"		if(src < m){																\n" \
"			int row_begin = row_offsets[src];												\n" \
"			int row_end = row_offsets[src + 1];												\n" \
"			for (int offset = row_begin; offset < row_end; ++ offset){									\n" \
"				int dst = column_indices[offset];											\n" \
"				float estimate = 0;													\n" \
"				for (int i = 0; i < 128; i++){												\n" \
"					estimate += user_lv[src*128+i] * item_lv[dst*128+i];								\n" \
"				}															\n" \
"				float error = rating[offset] - estimate;										\n" \
"				//*total_error += error * error;											\n" \
"				total += error * error;													\n" \
"																			\n" \
"			}																\n" \
"		}																	\n" \
"		//atomic_add(total_error,*total_error);													\n" \
"																			\n" \
"	union{																		\n" \
"		unsigned int intVal;															\n" \
"		float floatVal;																\n" \
"	}newVal;																	\n" \
"																			\n" \
"	union{																		\n" \
"		unsigned int intVal;															\n" \
"		float floatVal;																\n" \
"	}prevVal;																	\n" \
"																			\n" \
"	do{																		\n" \
"		prevVal.floatVal = *total_error;													\n" \
"		newVal.floatVal = prevVal.floatVal + total;												\n" \
"	}while(atomic_cmpxchg((volatile __global unsigned int*)total_error,										\n" \
"				prevVal.intVal, newVal.intVal)!=prevVal.intVal);									\n" \
"}																			\n" \
																			"\n" ;

void SGDSolver(int m, int n, int nnz, int *row_offsets, int *column_indices, ScoreT *rating, LatentT *user_lv, LatentT *item_lv, ScoreT lambda, ScoreT step, int *ordering, int max_iters, float epsilon) {
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

	//const char * slist[2] = {source, 0};
	//program = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	program_cal_delta = clCreateProgramWithSource(context, 1, (const char **) & kernelSource_cal_delta, NULL, &err);
	if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}

	
	program_cal_rmse = clCreateProgramWithSource(context, 1, (const char **) & kernelSource_cal_rmse, NULL, &err);
	if(err < 0){fprintf(stderr, "ERROR create program, err code: %d\n", err); exit(1);}
	
	err = clBuildProgram(program_cal_delta, 0, NULL, NULL, NULL, NULL);
	//if(err < 0){fprintf(stderr, "ERROR in build program, err code: %d\n", err); exit(1);}
	
	if(err < 0){
		size_t len;
		char buffer[1000];
	
		printf("error: build program delta failure\n");
		clGetProgramBuildInfo(program_cal_delta, device_id, CL_PROGRAM_BUILD_LOG,1000, buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}
	
	err = clBuildProgram(program_cal_rmse, 0, NULL, NULL, NULL, NULL);
	//if(err < 0){fprintf(stderr, "ERROR in build program, err code: %d\n", err); exit(1);}

	if(err < 0){
		size_t len;
		char buffer[1000];
	
		printf("error: build program rmse failure\n");
		clGetProgramBuildInfo(program_cal_rmse, device_id, CL_PROGRAM_BUILD_LOG,1000, buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	kernel_cal_delta = clCreateKernel(program_cal_delta, "sgd_kernel_cal_delta", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}
	
	kernel_cal_rmse = clCreateKernel(program_cal_rmse, "sgd_kernel_cal_rmse", &err);
	if(err < 0){fprintf(stderr, "ERROR in create kernel, err code: %d\n", err); exit(1);}

	float h_error;	
	h_error = 0.0;

	cl_mem d_row_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * (m+1), NULL, NULL);
	cl_mem d_column_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * nnz, NULL, NULL);
	cl_mem d_user_lv = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(LatentT) * m*K, NULL, NULL);
	cl_mem d_item_lv = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(LatentT) * n*K, NULL, NULL);
	cl_mem d_rating = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ScoreT) * nnz, NULL, NULL);
	cl_mem d_ordering = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * m, NULL, NULL);
	cl_mem total_error = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_row_offsets, CL_TRUE, 0, sizeof(int) * (m+1), row_offsets , 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_column_indices, CL_TRUE, 0, sizeof(int) * nnz, column_indices, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_user_lv, CL_TRUE, 0, sizeof(LatentT) * m*K, user_lv, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_item_lv, CL_TRUE, 0, sizeof(LatentT) * n*K, item_lv, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_rating, CL_TRUE, 0, sizeof(ScoreT) * nnz, rating, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_ordering, CL_TRUE, 0, sizeof(int) * m, ordering, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, total_error, CL_TRUE, 0, sizeof(float), &h_error, 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}

	
	err = clSetKernelArg(kernel_cal_delta, 0, sizeof(int), &m);
	err |= clSetKernelArg(kernel_cal_delta, 1, sizeof(int), &n);
	err |= clSetKernelArg(kernel_cal_delta, 2, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(kernel_cal_delta, 3, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(kernel_cal_delta, 4, sizeof(cl_mem), &d_rating);
	err |= clSetKernelArg(kernel_cal_delta, 5, sizeof(cl_mem), &d_user_lv);
	err |= clSetKernelArg(kernel_cal_delta, 6, sizeof(cl_mem), &d_item_lv);
	err |= clSetKernelArg(kernel_cal_delta, 7, sizeof(float), &lambda);
	err |= clSetKernelArg(kernel_cal_delta, 8, sizeof(float), &step);
	err |= clSetKernelArg(kernel_cal_delta, 9, sizeof(cl_mem), &d_ordering);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
	
	err = clSetKernelArg(kernel_cal_rmse, 0, sizeof(int), &m);
	err |= clSetKernelArg(kernel_cal_rmse, 1, sizeof(int), &n);
	err |= clSetKernelArg(kernel_cal_rmse, 2, sizeof(cl_mem), &d_row_offsets);
	err |= clSetKernelArg(kernel_cal_rmse, 3, sizeof(cl_mem), &d_column_indices);
	err |= clSetKernelArg(kernel_cal_rmse, 4, sizeof(cl_mem), &d_rating);
	err |= clSetKernelArg(kernel_cal_rmse, 5, sizeof(cl_mem), &d_user_lv);
	err |= clSetKernelArg(kernel_cal_rmse, 6, sizeof(cl_mem), &d_item_lv);
	err |= clSetKernelArg(kernel_cal_rmse, 7, sizeof(cl_mem), &total_error);
	if(err < 0){fprintf(stderr, "ERROR set kernel arg, err code: %d\n", err); exit(1);}
        //clFinish(queue);

	//for(int i = 0; i < 50; i++)
	//	printf("value of y-bef is %f\n", y[i]);
	
	//float h_error;	
	int iter = 0;
	Timer t;
	t.Start();


	
		err = clEnqueueNDRangeKernel(queue, kernel_cal_rmse, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}
		
		err = clEnqueueReadBuffer(queue, total_error, CL_TRUE, 0, sizeof(ScoreT), &h_error , 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}
		
		printf("iteration %d: RMSE error = %f per edge\n", iter, sqrt(h_error/nnz));

	do {
		++iter;
		h_error = 0.0;
		err = clEnqueueWriteBuffer(queue, total_error, CL_TRUE, 0, sizeof(ScoreT), &h_error , 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR write buffer, err code: %d\n", err); exit(1);}
	

		err = clEnqueueNDRangeKernel(queue, kernel_cal_delta, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}

		
		err = clEnqueueNDRangeKernel(queue, kernel_cal_rmse, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue nd range, err code: %d\n", err); exit(1);}


		
		err = clEnqueueReadBuffer(queue, total_error, CL_TRUE, 0, sizeof(ScoreT), &h_error , 0, NULL, NULL);
		if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}
		
		assert(h_error>0);
		printf("iteration %d: RMSE error = %f per edge\n", iter, sqrt(h_error/nnz));
	} while (iter < max_iters && h_error > epsilon);
	

	//t.Stop();

	clFinish(queue);
	
	t.Stop();
	
	err = clEnqueueReadBuffer(queue, d_user_lv, CL_TRUE, 0, sizeof(LatentT) * m * K, user_lv , 0, NULL, NULL);
	err |= clEnqueueReadBuffer(queue, d_item_lv, CL_TRUE, 0, sizeof(LatentT) * n * K, item_lv , 0, NULL, NULL);
	if(err < 0){fprintf(stderr, "ERROR enqueue read buffer, err code: %d\n", err); exit(1);}

	
	//for(int i = 0; i < 50; i++)
	//	printf("value of y-after is %f\n", y[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_row_offsets);
	clReleaseMemObject(d_column_indices);
	clReleaseMemObject(d_user_lv);
	clReleaseMemObject(d_item_lv);
	clReleaseMemObject(d_rating);
	clReleaseMemObject(d_ordering);
	clReleaseMemObject(total_error);
	clReleaseProgram(program_cal_delta);
	clReleaseProgram(program_cal_rmse);
	clReleaseKernel(kernel_cal_delta);
	clReleaseKernel(kernel_cal_rmse);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
	printf("\truntime [%s] = %f ms.\n", SGD_VARIANT, t.Millisecs());

	return;
}

