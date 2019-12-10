HOST_SRC:=../host-src/host.cpp
INCLUDES:=../host-includes

AOCL_COMPILE_CONFIG:=$(shell aocl compile-config)
AOCL_LINK_CONFIG:=$(shell aocl link-config)
AOCLFLAGS:= $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG)
CXXFLAGS:=$(CXXFLAGS) -O3 -std=c++11 -fopenmp
