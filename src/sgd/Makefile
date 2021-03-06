include ../common.mk
SRC=main.cc
OBJS=base.o omp_base.o main.o verifier.o
CXXFLAGS+=-std=c++11
NVFLAGS+=-std=c++11

all: $(OBJS) sgd_base sgd_omp_base sgd_omp_target sgd_vector sgd_ocl_base sgd_mic_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc sgd.h
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

sgd_base: base.o verifier.o ../../include/graph_io.h sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgd_vector: vector.o verifier.o ../../include/graph_io.h sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) verifier.o vector.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgd_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o main.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

sgd_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) ocl_base.o main.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

sgd_mic_base: mic_base.o main.o verifier.o
	$(ICC) $(CXXFLAGS) $(INCLUDES) mic_base.o main.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

sgd_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
