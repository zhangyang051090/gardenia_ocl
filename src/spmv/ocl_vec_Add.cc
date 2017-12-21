// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>


const char *kernelSource =																"\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable														\n"  \
"__kernel void vecAdd(__global double * a, __global double * b, __global double * c, const unsigned int n)	                                         \n" \
"{																			\n" \
"	int id = get_global_id(0);															\n" \
"	if(id < n)															   		\n" \
"	c[id]=a[id]+b[id];	   															\n" \
"}																			\n" \
																			"\n" ;


int main(int argc, char* argv[]){
	unsigned int n = 100000;

	double *h_a;
	double *h_b;
	double *h_c;

	cl_mem d_a;
	cl_mem d_b;
	cl_mem d_c;

	size_t bytes = n*sizeof(double);	
	
	h_a = (double*)malloc(bytes);
	h_b = (double*)malloc(bytes);
	h_c = (double*)malloc(bytes);
	
	for(int i = 0; i < n; i++)
	{
		h_a[i]=sinf(i)*sinf(i);
		h_b[i]=cosf(i)*cosf(i);
	
	}	

	cl_platform_id cpPlatform[32];
	cl_device_id device_id;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;

	cl_int err;
	size_t globalSize, localSize;
	
	localSize = 64;
	globalSize = ceil(n/(float)localSize)*localSize;
	

	//char *filechar = "/home/zy/gardinia_code/src/spmv/base.cl";
	//int sourcesize = 1024*1024;
	//char * source = (char *)calloc(sourcesize, sizeof(char));
	//FILE * fp = fopen(filechar, "rb");
	//fread(source + strlen(source), sourcesize, 1, fp);
	//fclose(fp);

	cl_uint num_platforms;
	char vendor[1024];
	cl_device_id devices[32];
	cl_uint num_devices;	

	clGetPlatformIDs(32, cpPlatform, &num_platforms);

	printf("\nNumber of platforms:\t%u\n\n", num_platforms);

	for(int i=0; i < num_platforms; i++)
		{
			clGetPlatformInfo(cpPlatform[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor,NULL);
			printf("\tPlatform Vendor:\t%s\n", vendor);
			
			clGetDeviceIDs(cpPlatform[i], CL_DEVICE_TYPE_ALL, sizeof(devices), devices, &num_devices);		
			printf("\tNumber of devices:\t%u\n\n", num_devices);	
		}

	



	err = clGetPlatformIDs(2, cpPlatform, NULL);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR platform id: %d", err); return -1;}	
	printf("platform id is %d\n", cpPlatform);
	
	//cl_platform_id platform;
	//cl_device_id *devices;
	char name_data[48];

//	cl_uint num_devices;
	err = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, NULL, &num_devices);
	if(err < 0){fprintf(stderr, "ERROR get num of devices, err code: %d\n", err); exit(1);}	
	printf("number of devices are %d\n", num_devices);

	//devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
	//clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU,num_devices, devices, NULL);
	
	//for(int i=0; i<num_devices; i++){
	//	err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name_data), name_data, NULL);
	//	if(err < 0){
	//		perror("couldn't read extension data");
	//		exit(1);
	//		}
	//}
	//free(devices);

	
	err = clGetDeviceIDs(cpPlatform[1], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR: %d\n", err); return -1;}	
	
	printf("device_id %d\n", device_id);


	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR1: %d\n", err); return -1;}	
	queue = clCreateCommandQueue(context, device_id, 0, &err);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR2: %d\n", err); return -1;}	

	//const char * slist[2] = {source, 0};
	//program = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR3: %d\n", err); return -1;}	
	clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR4: %d\n", err); return -1;}	
	kernel = clCreateKernel(program, "vecAdd", &err);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR5: %d\n", err); return -1;}	

	
	d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
	d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

	err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR6: %d\n", err); return -1;}	
	err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR7: %d\n", err); return -1;}	

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR8: %d\n", err); return -1;}	
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR9: %d\n", err); return -1;}	
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR10: %d\n", err); return -1;}	
	err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR11: %d\n", err); return -1;}	
	

	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	if(err != CL_SUCCESS){fprintf(stderr, "ERROR12: %d\n", err); return -1;}	

	clFinish(queue);
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
	
	//for(int i = 0; i < n/100;i ++)
	//	printf("%f ",h_c[i]);

	double sum = 0;
	for(int i = 0; i < n;i ++)
		sum += h_c[i];

	printf("final result: %f\n", sum/n);

//	for(int i = 0; i < 50; i++)
//		printf("value of y-after is %f\n", y[i]);
	
//	printf("finished results");
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	free(h_a);
	free(h_b);
	free(h_c);
	return 0;
}

