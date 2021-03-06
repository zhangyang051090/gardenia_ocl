include ../common.mk
SRC=main.cc
OBJS=base.o tex.o warp.o vector.o omp_base.o verifier.o main.o

all: $(OBJS) spmv_base spmv_vector spmv_omp_base spmv_omp_target spmv_ocl_base spmv_mic_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -march=core-avx2 -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

spmv_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

spmv_mic_base: mic_base.o main.o verifier.o
	$(ICC) $(CXXFLAGS) $(INCLUDES) main.o mic_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

spmv_omp_simd: omp_simd.o main.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) main.o omp_simd.o verifier.o -o $@
	mv $@ $(BIN)

spmv_base: main.o base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_tex: main.o tex.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o tex.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_warp: main.o warp.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o warp.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_vector: main.o vector.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
