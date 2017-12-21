HOME=/home/zy
CUDA_HOME=/usr/local/cuda
ICC_HOME=/opt/intel/composer_xe_2015.1.133/bin/intel64
GARDINIA_ROOT=$(HOME)/gardinia_code
CUB_DIR=$(HOME)/cub-1.6.4
VEX_DIR=$(HOME)/vexcl-master
B40_DIR=$(HOME)/back40computing-read-only
BIN=$(GARDINIA_ROOT)/bin
CC=gcc
CXX=g++
ICC=$(ICC_HOME)/icc
ICPC=$(ICC_HOME)/icpc
NVCC=$(CUDA_HOME)/bin/nvcc
COMPUTECAPABILITY=sm_37
CXXFLAGS=-O3 -Wall -fopenmp
ICPCFLAGS=-O3 -Wall -openmp
#NVFLAGS=-g -arch=$(COMPUTECAPABILITY) #-Xptxas -v
NVFLAGS=-O3 -arch=$(COMPUTECAPABILITY) -Wno-deprecated-gpu-targets#-Xptxas -v
#NVFLAGS+=-cudart shared
INCLUDES=-I$(CUDA_HOME)/include -I$(GARDINIA_ROOT)/include
#INCLUDES=-I/opt/intel/opencl-1.2-6.4.0.25 -I$(GARDINIA_ROOT)/include
LIBS=-L$(CUDA_HOME)/lib64 -L/opt/intel/composer_xe_2015.1.133/compiler/lib/mic
#LIBS=-L/opt/intel/composer_xe_2015.1.133/compiler/lib/mic
