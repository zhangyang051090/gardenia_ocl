include ../common.mk
INCLUDES+=-I $(VEX_DIR)
ICPCFLAGS+=-std=c++98
#CXXFLAGS+=-std=c++98
B40C_INC=-I $(B40_DIR) -I $(B40_DIR)/test
SRC=main.cc
OBJS= hello.o

all: $(OBJS) hello

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<


hello: hello.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o hello hello.cpp $@ -lOpenCL
	mv $@ $(BIN)

clean:
	rm *.o
