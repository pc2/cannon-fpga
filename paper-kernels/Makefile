include ../common.mk

host: $(HOST_SRC) 
	$(CXX) $(CXXFLAGS) -I$(INCLUDES) $(AOCLFLAGS) $< -o $@ 

host-fastemu: $(HOST_SRC) 
	$(CXX) $(CXXFLAGS) -DFAST_EMU -I$(INCLUDES) -lOpenCL $< -o $@ 

clean:
	$(RM) host host-fastemu

