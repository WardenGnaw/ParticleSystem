CC=g++
CPPSOURCES=SerialUpdate.cpp ParticleSystem.cpp Fluid.cpp GLSL_helper.cpp GeometryCreator.cpp MStackHelp.cpp
CPPFLAGS= -DGL_GLEXT_PROTOTYPES 
APPLE_CPPFLAGS= -framework OpenGL -framework GLUT -Wno-deprecated -Wno-parentheses
LINUX_CPPFLAGS= -lGL -lGLU -lglut
CU_FLAGS= -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -O3


NCC=nvcc
CUSOURCES=ParticleUpdate.cu

all: linux

mac:
	$(NCC) $(CUSOURCES) $(CU_FLAGS) $(CPPSOURCES) $(CPPFLAGS) $(APPLE_CPPFLAGS) -DENABLE_CUDA

linux:
	$(NCC) $(CUSOURCES) $(CU_FLAGS) $(CPPSOURCES) $(CPPFLAGS) $(LINUX_CPPFLAGS) -DENABLE_CUDA

serLinux:
	$(NCC) $(CUSOURCES) $(CU_FLAGS) $(CPPSOURCES) $(CPPFLAGS) $(LINUX_CPPFLAGS)
