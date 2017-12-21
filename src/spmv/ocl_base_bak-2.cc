// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "spmv.h"
#include <CL/cl.h>
#include <string.h>
#include "timer.h"
#define SPMV_VARIANT "ocl_base"

// local variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

int initialize(int use_gpu) {
    cl_int result;
    size_t size;
    // create OpenCL context
    cl_platform_id platform_id[32];
    if (clGetPlatformIDs(2, platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
    cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0};
    device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
    context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
    if( !context ) { fprintf(stderr, "ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }
    // get the list of GPUs
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
    num_devices = (int) (size / sizeof(cl_device_id));
    printf("num_devices = %d\n", num_devices);
    if( result != CL_SUCCESS || num_devices < 1 ) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }
    device_list = new cl_device_id[num_devices];
    if( !device_list ) { fprintf(stderr, "ERROR: new cl_device_id[] failed\n"); return -1; }
    result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
    if( result != CL_SUCCESS ) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }
    // create command queue for the first device
    cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
    if( !cmd_queue ) { fprintf(stderr, "ERROR: clCreateCommandQueue() failed\n"); return -1; }
    return 0;
}

void SpmvSolver(int num_rows, int nnz, int *Ap, int *Aj, ValueType *Ax, ValueType *x, ValueType *y) {
	printf("Launching OpenCL SpMV solver ...\n");

	//load OpenCL kernel file
	cl_int err = 0;
	char *filechar = "/home/zy/gardinia_code/src/spmv/base.cl";
	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char));
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return; }
	FILE * fp = fopen(filechar, "rb");
	if(!fp) { printf("ERROR: unable to open '%s'\n", filechar); return; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

	// OpenCL initialization
	if(initialize(1)) return;

	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return; }
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);

	{ // show warnings/errors
      static char log[65536]; 
      memset(log, 0, sizeof(log));
      cl_device_id device_id = 0;
      //get context info
      err = clGetContextInfo(context, 
                             CL_CONTEXT_DEVICES, 
                             sizeof(device_id), 
                             &device_id, 
                             NULL);
      //get program build info
      clGetProgramBuildInfo(prog, 
                            device_id, 
                            CL_PROGRAM_BUILD_LOG, 
                            sizeof(log)-1, 
                            log, 
                            NULL);
							
      if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
  	 }

	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return; }

	
	//zy
	char * kernel_1 = "spmv_kernel";
	cl_kernel spmv_kernel_1=clCreateKernel(prog, kernel_1, &err);
	if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() => %d\n", err); return; }
	//std:vector<float> Ap(num_rows+1),Aj(nnz),Ax(nnz),x(nnz),y(num_rows)
	//int Ap[num_rows+1],Aj[nnz];
	//ValueType Ax[nnz],x[nnz],y[num_rows];
	//for(int i=0;i < DATA_SIZE;i++){
	//	a[i] = i;
	//	b[i] = i;	
	//}
	

	//cl_mem cl_Ap = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(cl_float) * (num_rows+1), Ap, NULL);
	//cl_mem cl_Aj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(cl_float) * nnz, Aj, NULL);
	//cl_mem cl_Ax = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(cl_float) * nnz, Ax, NULL);
	//cl_mem cl_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(cl_float) * nnz, x, NULL);
	//cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY,sizeof(cl_float) * num_rows, NULL, NULL);
	
	Timer t;
	t.Start();

	//create the device-side graph structure
	cl_mem cl_Ap = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * (num_rows+1), NULL, &err);
	if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer ap (size:%d) => %d\n",  num_rows+1 , err); return;}

	cl_mem cl_Aj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * nnz, NULL, &err);
	if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer aj (size:%d) => %d\n",  nnz , err); return;}

	cl_mem cl_Ax = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ValueType) * nnz, NULL, &err);
	if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer ax (size:%d) => %d\n", nnz , err); return;}

	cl_mem cl_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ValueType) * nnz, NULL, &err);
	if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer x (size:%d) => %d\n",  nnz , err); return;}

	cl_mem cl_y = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(ValueType) * num_rows, NULL, &err);
	if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer res (size:%d) => %d\n",  num_rows , err); return;}


    //copy data to device side buffers
    err = clEnqueueWriteBuffer(cmd_queue, 
                               cl_Ap, 
                               1, 
                               0, 
                               (num_rows + 1) * sizeof(int), 
                               Ap, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer ap (size:%d) => %d\n", num_rows, err); return; }

 err = clEnqueueWriteBuffer(cmd_queue, 
                               cl_Aj, 
                               1, 
                               0, 
                               nnz * sizeof(int), 
                               Aj, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer aj (size:%d) => %d\n", nnz, err); return; }

 err = clEnqueueWriteBuffer(cmd_queue, 
                               cl_Ax, 
                               1, 
                               0, 
                               nnz * sizeof(ValueType), 
                               Ax, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer ax (size:%d) => %d\n", nnz, err); return; }

 err = clEnqueueWriteBuffer(cmd_queue, 
                               cl_x, 
                               1, 
                               0, 
                               nnz* sizeof(ValueType), 
                               x, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer x (size:%d) => %d\n", nnz, err); return; }

 err = clEnqueueWriteBuffer(cmd_queue, 
                               cl_y, 
                               1, 
                               0, 
                               num_rows * sizeof(ValueType), 
                               y, 
                               0, 
                               0, 
                               0);
							   
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer y (size:%d) => %d\n", num_rows, err); return; }
	printf(" xxxxxxxxxxxxxxxxxxxx");
	//set up the kernel arguments
	clSetKernelArg(spmv_kernel_1, 0, sizeof(cl_int), &num_rows);
	clSetKernelArg(spmv_kernel_1, 1, sizeof(cl_mem), &cl_Ap);
	clSetKernelArg(spmv_kernel_1, 2, sizeof(cl_mem), &cl_Aj);
	clSetKernelArg(spmv_kernel_1, 3, sizeof(cl_mem), &cl_Ax);
	clSetKernelArg(spmv_kernel_1, 4, sizeof(cl_mem), &cl_x);
	clSetKernelArg(spmv_kernel_1, 5, sizeof(cl_mem), &cl_y);

	//zy	
	//size_t num_rows_convert=0;
	//num_rows_convert = static_cast<size_t>(num_rows);
	int block_size = num_rows;
	size_t global_work[3] = {block_size, 1, 1};

	printf(" zyyyyyyyyyyyyyyyyyyyyyyy");
	//launch the kernel
	err = clEnqueueNDRangeKernel(cmd_queue, spmv_kernel_1, 1, NULL, global_work, NULL, 0, 0, 0);
	  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel  clEnqueueNDRangeKernel()=>%d failed\n", err); return; }
	printf(" zzzzzzzzzzzzzzzzzzzzzzy");
	//zy??
	//ValueType y[num_rows];
	err = clEnqueueReadBuffer(cmd_queue, cl_y, CL_TRUE, 0, sizeof(ValueType) * num_rows, y, 0, 0, 0);

	//zy release
	clReleaseKernel(spmv_kernel_1);
	clReleaseProgram(prog);
	clReleaseMemObject(cl_Ap);
	clReleaseMemObject(cl_Aj);
	clReleaseMemObject(cl_Ax);
	clReleaseMemObject(cl_x);
	clReleaseMemObject(cl_y);
	
	//zy
	for(size_t i = 0; i < num_devices; i ++){
	clReleaseCommandQueue(cmd_queue);
	}
	clReleaseContext(context);
	
	t.Stop();
	printf("\truntime [%s] = %f ms.\n", SPMV_VARIANT, t.Millisecs());
	return;
}
