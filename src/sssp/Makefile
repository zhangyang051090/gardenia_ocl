include ../common.mk
INCLUDES+=-I $(CUB_DIR)
#ICPCFLAGS+=-std=c++98
EXE=sssp
SRC=main.cc
OBJS= topo_base.o topo_lb.o linear_base.o linear_lb.o main.o verifier.o omp_base.o

all: $(OBJS) $(EXE)_linear_base $(EXE)_linear_lb $(EXE)_topo_base $(EXE)_topo_lb $(EXE)_omp_base $(EXE)_omp_target $(EXE)_ocl_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

$(EXE)_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

$(EXE)_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

$(EXE)_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@
	mv $@ $(BIN)

$(EXE)_linear_base: main.o linear_base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_linear_lb: main.o linear_lb.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_dstep: main.o dstep.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o dstep.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_topo_base: main.o topo_base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_topo_lb: main.o topo_lb.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

