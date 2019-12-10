/*
 MIT License
 
 Copyright (c) 2019 Paolo Gorlani 
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE. 
*/

#include <CL/cl.hpp>

#ifdef FAST_EMU
  #define CL_CHANNEL_1_INTELFPGA 0
  #define CL_CHANNEL_2_INTELFPGA 0
  #define CL_CHANNEL_3_INTELFPGA 0
  #define CL_CHANNEL_4_INTELFPGA 0
  #define CL_MEM_HETEROGENEOUS_INTELFPGA 0
#else
  #include <CL/cl_ext_intelfpga.h>
#endif

#include <iostream>
#include <iomanip>
#include <fstream>

#include <vector>
#include <string>

#include <algorithm>
#include <limits>
#include <cmath>

#include <float_classifier.hpp>

#define PLATFORM_ID 0
#define DEVICE_ID 0

unsigned int float_to_uint(float);
unsigned int float_ulp_distance(float, float);

int main(int argc, char* argv[])
{

#ifdef EXTRA_ARGS

  if(argc != 5)
  {
    std::cout<<"Wrong number of parameters!\n";
    std::cout<<"Usage: "<<argv[0]<<" <#CU> <block-dim> <dim> <.aocx file path>\n";
    std::cout<<"  <#CU> is the number of compute units (i.e. synthesized kernels)\n";
    std::cout<<"  <block-dim> is the block size of the synthesized kernels (i.e. dim1)\n";
    std::cout<<"  <dim> is the size of the matrices, must be a multiple of <block-dim>"<<std::endl;
    return EXIT_FAILURE;
  }

  const size_t num_kernels = std::atoi(argv[1]);
  const size_t dim1 = std::atoi(argv[2]); 
  const size_t dim = std::atoi(argv[3]);
  const char* aocx_filename = argv[4];

#else

  if(argc != 4)
  {
    std::cout<<"Wrong number of parameters!\n";
    std::cout<<"Usage: "<<argv[0]<<" <#CU> <dim> <.aocx file path>\n";
    std::cout<<"  <#CU> is the number of compute units (i.e. synthesized kernels)\n";
    std::cout<<"  <dim> is the size of the matrices\n";
    std::cout<<"NOTE: <dim> must be a multiple of the block dimension (dim1) in the synthesized kernel code."<<std::endl;
    return EXIT_FAILURE;
  }

  const size_t num_kernels = std::atoi(argv[1]);
  const size_t dim = std::atoi(argv[2]);
  const char* aocx_filename = argv[3];

#endif

  const size_t LENGTH = dim*dim;

  const size_t buffer_size = sizeof(float)*LENGTH; 
  float *source_a, *source_b, *sp_fpga_result;

  int err;
  err  = posix_memalign((void**)&source_a, 64*sizeof(float), buffer_size);
  err += posix_memalign((void**)&source_b, 64*sizeof(float), buffer_size);
  err += posix_memalign((void**)&sp_fpga_result, 64*sizeof(float), buffer_size);

  if(err)
  {
    std::cout<<"Error: host memory allocation." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout<<"\n Matrix multiplication sizes"<<std::endl;
  std::cout<<"  dim  = "<<dim<<std::endl;
  std::cout<<"  size of matrices = ("<<dim<<", "<<dim<<")"<<std::endl;

#ifdef EXTRA_ARGS
  std::cout<<"  dim1 = "<<dim1<<std::endl;
  std::cout<<"  size of blocks = ("<<dim1<<", "<<dim1<<")"<<std::endl;
  
  if(dim%dim1)
  {
    std::cout<<"Error: the matrix size (dim) needs to be a multiple the block size (dim1)." << std::endl;
    return EXIT_FAILURE;
  }
#endif

#ifdef CHECK_WITH_ONES

  std::vector<double> dp_host_result (LENGTH, dim);

  #pragma omp parallel for
  for(size_t i=0; i < LENGTH; i++)
  {
    source_a[i] = 1; 
    source_b[i] = 1;
  }

  std::cerr<<"\nMatrices are filled with ones, result = "<<dim<<std::endl;
 
#else

  std::cerr<<"\nComputing matrix multiplication on host (it can take long time) ... ";

  const float normalize = 1.0f/float(RAND_MAX);

  std::srand(9);

  for(size_t i=0; i < LENGTH; i++)
  {
    source_a[i] = normalize*float(rand());
    source_b[i] = normalize*float(rand());
  }

  std::vector<double> dp_host_result (LENGTH, 0.0);

  #pragma omp parallel for
  for(size_t i=0; i < dim; i++)
    for(size_t k=0; k < dim; k++)
      for(size_t j=0; j < dim; j++)
        dp_host_result[i*dim+j] += source_a[i*dim+k]*source_b[k*dim+j]; 

  std::cerr<<"done!"<<std::endl;

#endif

  // OpenCL host code starts --------------------------------------------------

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  std::cout<<"\nSelecting OpenCL device (PLATFORM_ID="
           <<PLATFORM_ID<<", DEVICE_ID="<<DEVICE_ID<<")"
           <<std::endl;

  cl::Platform platform = platforms[PLATFORM_ID];
  std::cout<<"  Platform: "<<platform.getInfo<CL_PLATFORM_NAME>()<<std::endl;

  std::vector<cl::Device> devices;
  platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);

  cl::Device device = devices[DEVICE_ID];
  std::cout<<"  Device: "<<device.getInfo<CL_DEVICE_NAME>()<<std::endl;

  cl::Context context(devices);

  std::vector<cl::CommandQueue> q;

  for(size_t i=0; i<num_kernels; ++i)
    q.push_back(cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE));

  std::cout<<"\nLoading bitstream form "<<aocx_filename<<" ..."<<std::endl;
  std::ifstream bin_file(aocx_filename, std::ifstream::binary);
  if(bin_file.fail())
  {
    std::cout<<"Error reading "<<aocx_filename<<std::endl;
    return EXIT_FAILURE;
  }

  bin_file.seekg (0, bin_file.end);
  size_t nb = bin_file.tellg();
  bin_file.seekg (0, bin_file.beg);

  char *buf = new char [nb];
  bin_file.read(buf, nb);


  cl::Program::Binaries bins;
  bins.push_back({buf,nb});

  devices.resize(1);
  cl::Program program(context, devices, bins);
  program.build();

  std::cout<<"\nKernel creation"<<std::endl;
  
  std::vector<cl::Kernel> kernels;
  std::vector<std::string> kernel_names;

  for(size_t i=0; i<num_kernels; ++i)
  {
    std::string kernel_name("krnl_cannon_"+std::to_string(i));

    int err;
    kernels.push_back(cl::Kernel(program, kernel_name.c_str(), &err));
    if (err != CL_SUCCESS)
    {
      std::cout<<"  "<<kernel_name<<": KO"<<std::endl;
      std::cout<<"Error in kernel creation."<<std::endl;
      return EXIT_FAILURE;
    }
    else std::cout<<"  "<<kernel_name<<": OK"<<std::endl;
    kernel_names.push_back(kernel_name);
  }

  cl::Buffer buffer_a(context, CL_MEM_READ_ONLY|CL_CHANNEL_1_INTELFPGA|CL_MEM_HETEROGENEOUS_INTELFPGA,  buffer_size);
  cl::Buffer buffer_b(context, CL_MEM_READ_ONLY|CL_CHANNEL_2_INTELFPGA|CL_MEM_HETEROGENEOUS_INTELFPGA,  buffer_size);
  cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY|CL_CHANNEL_3_INTELFPGA|CL_MEM_HETEROGENEOUS_INTELFPGA, buffer_size);

  std::vector<cl::Event> kernel_event(num_kernels);

#ifdef EXTRA_ARGS
  const unsigned int NB2 = dim/dim1;
  const unsigned int CHUNCK = NB2/num_kernels; 
  const unsigned int R = NB2%num_kernels;
#endif

  std::cout<<"\nSet arguments"<<std::endl;
  const unsigned int _dim = dim;
  for(unsigned int i=0; i<num_kernels; ++i)
  {
    int err;
    err = kernels[i].setArg(0, buffer_a);
    err += kernels[i].setArg(1, buffer_b);
    err += kernels[i].setArg(2, buffer_c);
    err += kernels[i].setArg(3, _dim);
    std::cout<<"  krnl_cannon_"<<i<<":\t";

#ifdef EXTRA_ARGS
    const unsigned int _I2start = std::min(i*(CHUNCK)+((i<R) ? i : R),NB2); 
    const unsigned int _I2end = std::min((i+1)*(CHUNCK)+((i+1<R) ? i+1 : R),NB2);
    std::cout<<_I2start<<"\t"<<_I2end<<"\t"<<_I2end-_I2start;
    err += kernels[i].setArg(4, _I2start);
    err += kernels[i].setArg(5, _I2end);
#endif

    if (err != CL_SUCCESS)
    {
      std::cout<<"\tKO"<<std::endl;
      std::cout<<"Error in setting kernel arguments."<<std::endl;
      return EXIT_FAILURE;
    } else std::cout<<"\tOK"<<std::endl;
   
  } 

  q[0].enqueueWriteBuffer(buffer_a, CL_TRUE, 0, buffer_size, source_a);
  q[0].enqueueWriteBuffer(buffer_b, CL_TRUE, 0, buffer_size, source_b);

  q[0].finish();
  std::cout<<"\nKernel execution start"<<std::endl;

  for(size_t i=0; i<num_kernels; ++i)
    q[i].enqueueNDRangeKernel(kernels[i],cl::NullRange,cl::NDRange(1),cl::NDRange(1),NULL,&(kernel_event[i]));

  for(size_t i=0; i<num_kernels; ++i)
    q[i].finish();
  std::cout<<"Kernel execution finish"<<std::endl;

  q[0].enqueueReadBuffer(buffer_c, CL_TRUE, 0, buffer_size, sp_fpga_result);

  std::cout<<"\nKernel times [ns]"<<std::endl;

  std::vector<unsigned long> start_times;
  std::vector<unsigned long> end_times;

  for(size_t i=0; i<num_kernels; ++i)
  {
    start_times.push_back(kernel_event[i].getProfilingInfo<CL_PROFILING_COMMAND_START>());
    end_times.push_back(kernel_event[i].getProfilingInfo<CL_PROFILING_COMMAND_END>());
    std::cout<<"  "<<kernel_names[i]
             <<": start="<<start_times[i]<<"\tstop="<<end_times[i]
             <<"\telapsed="<<end_times[i]-start_times[i]<<std::endl;
  }

  unsigned long elapsed_ns = *std::max_element(end_times.begin(),end_times.end())
                             - *std::min_element(start_times.begin(), start_times.end());

  // OpenCL host code ends ----------------------------------------------------

  // Result Check

  std::vector<size_t> ulp_histogram(16,0); 

  double max_rel_err = 0;
  unsigned long max_ulp = 0;
  unsigned long min_ulp = std::numeric_limits<unsigned long>::max();
  unsigned long mean_ulp = 0;

  std::vector<double> v_rel_err;
  std::vector<unsigned long> v_ulp;
  bool mean_ulp_wrap = false;    

  float_classifier<float> _fpclass;

  for(size_t i = 0; i < LENGTH; i++) 
  {
    _fpclass.eval(sp_fpga_result[i]);

    double ferr = std::abs((float(dp_host_result[i]) - sp_fpga_result[i])/float(dp_host_result[i]));
    unsigned long ulp = float_ulp_distance(float(dp_host_result[i]), sp_fpga_result[i]);
    if(ulp < ulp_histogram.size()) ++ulp_histogram[ulp];

    v_rel_err.push_back(ferr);
    v_ulp.push_back(ulp);

    max_ulp = std::max(max_ulp, ulp);
    min_ulp = std::min(min_ulp, ulp);
    max_rel_err = std::max(max_rel_err, ferr);
    mean_ulp_wrap |= (mean_ulp + ulp < mean_ulp);
    mean_ulp += ulp;
  }

  std::cout<<"\nResult check"<<std::endl;
  std::cout<<"\n  Floating-point class output summary"<<std::endl;
  const double perc_tot = 100.0/_fpclass.total();
  std::cout<<"    #NaNs:\t"<<_fpclass.nans()<<" ("<<perc_tot*_fpclass.nans()<<" %)"<<std::endl;
  std::cout<<"    #infs:\t"<<_fpclass.infs()<<" ("<<perc_tot*_fpclass.infs()<<" %)"<<std::endl;
  std::cout<<"    #normals:\t"<<_fpclass.normals()<<" ("<<perc_tot*_fpclass.normals()<<" %)"<<std::endl;
  std::cout<<"    #subnormals:\t"<<_fpclass.subnormals()<<" ("<<perc_tot*_fpclass.subnormals()<<" %)"<<std::endl;
  std::cout<<"    #zeros:\t"<<_fpclass.zeros()<<" ("<<perc_tot*_fpclass.zeros()<<" %)"<<std::endl;

  auto max_rel_err0 = std::max_element(v_rel_err.begin(), v_rel_err.end());
  size_t pos_max_rel_err0 = max_rel_err0 - v_rel_err.begin();
  auto max_ulp0 = std::max_element(v_ulp.begin(), v_ulp.end());
  size_t pos_max_ulp0 = max_ulp0 - v_ulp.begin(); 

  const int max_precision = std::numeric_limits<float>::digits10 + 1;
 
  std::cout<<"\n  Correctness"<<std::endl;

  std::cout<<"    Max ULP distance: "<<*max_ulp0
            <<"  at position: ("<<pos_max_ulp0/dim<<", "<<pos_max_ulp0%dim<<")"
            <<"  host: "<<std::setprecision(max_precision)<<float(dp_host_result[pos_max_ulp0])
            <<"  fpga: "<<std::setprecision(max_precision)<<sp_fpga_result[pos_max_ulp0]
            <<std::endl;

  std::cout<<"    ULP distance: min("<<min_ulp<<") max("<<max_ulp<<") mean("<<double(mean_ulp)/double(LENGTH)<<")";
  if(mean_ulp_wrap) std::cout<<"[ERROR mean ulp wrapped!]";
  std::cout<<std::endl; 

  std::cout<<"\n    Max relative error: "<<*max_rel_err0
             <<"  at position: ("<<pos_max_rel_err0/dim<<", "<<pos_max_rel_err0%dim<<")"
             <<"  host: "<<std::setprecision(max_precision)<<float(dp_host_result[pos_max_rel_err0])
             <<"  fpga: "<<std::setprecision(max_precision)<<sp_fpga_result[pos_max_rel_err0]
             <<std::endl;

#ifdef PRINT_ULP_HIST
  std::cout<<"\n  ULP histogram"<<std::endl;
  for(size_t i=0; i<ulp_histogram.size(); ++i)
    std::cout<<"    "<<i<<"  ULP:\t"<<ulp_histogram[i]<<"\t("<<100.0*double(ulp_histogram[i])/double(LENGTH)<<"%)"<<std::endl; 
#endif

  std::cout<<"\nKernels performances"<<std::endl;
  const size_t ops = size_t(dim)*size_t(dim)*size_t(dim+dim-1); // (dim) multiplications + (dim-1) add for each matrix element 
  const double GFLOPS = double(ops)/double(elapsed_ns); 
  std::cout<<"  Overall kernel execution time: "<<elapsed_ns<<" ns"<<std::endl;
  std::cout<<"  Overall kernel floating-point performance: "<<GFLOPS<<" GFLOPS"<<std::endl;


  delete buf;
  free(source_a);
  free(source_b);
  free(sp_fpga_result);

  return 0;

}

unsigned int float_to_uint(float f_a)
{

  static_assert(sizeof(float) == sizeof(unsigned int), "unsigned int/float sizes differ"); 

  unsigned int u_a;
  std::memcpy(&u_a, &f_a, sizeof(float));

  return u_a; 
}

unsigned int float_ulp_distance(float f_a, float f_b)
{

  static_assert(sizeof(float) == sizeof(unsigned int), "unsigned int/float sizes differ"); 

  unsigned int ulp;
  unsigned int a, b;

  std::memcpy(&a, &f_a, sizeof(float));
  std::memcpy(&b, &f_b, sizeof(float));

  if (a > b)
    ulp = a - b;
    else
      ulp = b - a;

  return ulp; 
}

