include ../common.mk

host: $(HOST_SRC) 
	$(CXX) $(CXXFLAGS) -DEXTRA_ARGS -I$(INCLUDES) $(AOCLFLAGS) $< -o $@ 

host-fastemu: $(HOST_SRC) 
	$(CXX) $(CXXFLAGS) -DEXTRA_ARGS -DFAST_EMU -I$(INCLUDES) -lOpenCL $< -o $@ 

clean:
	$(RM) host host-fastemu

