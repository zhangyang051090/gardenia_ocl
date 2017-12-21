include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cc
OBJS=omp_base.o base.o warp.o main.o verifier.o
all: $(OBJS) tc_base tc_warp tc_omp_base tc_omp_target tc_ocl_base tc_mic_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

tc_base: base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_warp: warp.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o warp.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o main.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

tc_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) ocl_base.o main.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

tc_mic_base: mic_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) mic_base.o main.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

tc_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
