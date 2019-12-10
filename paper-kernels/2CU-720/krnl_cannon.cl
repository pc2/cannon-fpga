#define dim0 8          
#define dim1 720 
#define NB1 (dim1/dim0)

#define __fpga_reg2(x) __fpga_reg(__fpga_reg(x)) 
#define __fpga_reg4(x) __fpga_reg(__fpga_reg(__fpga_reg(__fpga_reg(x))))

inline void
cannon_func( const uint dim2, const uint I2begin, const uint I2end, 
             __global volatile const float * restrict gA, 
             __global volatile const float * restrict gB, 
             __global volatile float * restrict gC )
{

  const uint NB2 = dim2/dim1;

  #pragma loop_coalesce 2
  for(uint I2=I2begin; I2<I2end; ++I2)
    #pragma max_concurrency 1
    for(uint J2=0; J2<NB2; ++J2)
    {
     
      float C1[NB1][NB1][dim0][dim0] __attribute((numbanks(dim0*dim0))); 
      
      #pragma max_concurrency 1
      for(uint K2=0; K2<NB2; ++K2)
      {
      
        float A1[NB1][NB1][dim0][dim0] __attribute((numbanks(dim0*dim0)));
        float B1[NB1][NB1][dim0][dim0] __attribute((numbanks(dim0*dim0)));
    
#if 0  
    
        #pragma loop_coalesce 2
        for(uint i1=0; i1<dim1; ++i1)
          #pragma unroll dim0 
          for(uint j1=0; j1<dim1; ++j1)
          {
            const ulong iA = dim2*( (ulong)(I2*dim1) + i1 ) + (ulong)(K2*dim1) + j1;
            const ulong iB = dim2*( (ulong)(K2*dim1) + i1 ) + (ulong)(J2*dim1) + j1;

            A1[i1/dim0][j1/dim0][i1%dim0][j1%dim0] = __fpga_reg2(gA[iA]);
            B1[j1/dim0][i1/dim0][i1%dim0][j1%dim0] = __fpga_reg2(gB[iB]);
          }
    
#else
    
        #pragma loop_coalesce 2
        for(uint i1=0; i1<dim1; ++i1)
        {
   
          #pragma unroll dim0 
          for(uint j1=0; j1<dim1; ++j1)
          {

            float volatile _pA[dim0] __attribute((register));
            float volatile _pB[dim0] __attribute((register));
 
            const ulong iA = dim2*( (ulong)(I2*dim1) + i1 ) + (ulong)(K2*dim1) + j1;
            const ulong iB = dim2*( (ulong)(K2*dim1) + i1 ) + (ulong)(J2*dim1) + j1;
    
            _pA[i1%dim0] = __fpga_reg2(gA[iA]);
            _pB[i1%dim0] = __fpga_reg2(gB[iB]);
    
            #pragma unroll dim0 
            for(uchar _i=0; _i<dim0; ++_i)
              if(_i == i1%dim0)
              {      
                A1[i1/dim0][j1/dim0][_i][j1%dim0] = __fpga_reg2(_pA[_i]);
                B1[j1/dim0][i1/dim0][_i][j1%dim0] = __fpga_reg2(_pB[_i]);
              }
          }

        }
    
#endif
      
        #pragma loop_coalesce 2 
        for(uchar I1=0; I1<NB1; ++I1)
          for(uchar J1=0; J1<NB1; ++J1)
          {
      
            float C0[dim0][dim0] = {0.0f};

            for(uchar K1=0; K1<NB1; ++K1)
            {
      
              float A0[dim0][dim0+1];
              float B0[dim0+1][dim0];
        
              #pragma unroll dim0
              for(uchar i=0; i<dim0; ++i)
                #pragma unroll dim0
                for(uchar j=0; j<dim0; ++j)
                {
                  A0[i][(j+dim0-i)%dim0] = A1[I1][K1][i][j];
                  B0[(i+dim0-j)%dim0][j] = B1[J1][K1][i][j];  // reversed index
                }
      
              #pragma unroll dim0
              for(uchar k=0; k<dim0; ++k)
              {
                 
                #pragma unroll dim0
                for(uchar z=0; z<dim0; ++z)
                {
                  A0[z][dim0] = A0[z][0];
                  B0[dim0][z] = B0[0][z];   
                }
                    
                #pragma unroll dim0
                for(uchar i=0; i<dim0; ++i)
                  #pragma unroll dim0
                  for(uchar j=0; j<dim0; ++j)
                  {
                    C0[i][j] += __fpga_reg(A0[i][j]) * __fpga_reg(B0[i][j]);
                    A0[i][j] = __fpga_reg(A0[i][j+1]); 
                    B0[i][j] = __fpga_reg(B0[i+1][j]);
                  }
                    
              }
                   
             } //K1 
    
    
             #pragma unroll dim0
             for(uchar i=0; i<dim0; ++i)
               #pragma unroll dim0
               for(uchar j=0; j<dim0; ++j)
                 C1[I1][J1][i][j] = ( K2 ? C1[I1][J1][i][j] : 0.0f ) + C0[i][j];
    
          } // I1 J1 
    
      } //K2
   
#if 0    

      #pragma loop_coalesce 2
      for(uint i1=0; i1<dim1; ++i1)
        #pragma unroll dim0 
        for(uint j1=0; j1<dim1; ++j1)
        {
          const ulong iC = dim2*( (ulong)(I2*dim1) + i1 ) + (ulong)(J2*dim1) + j1;
          gC[iC]=__fpga_reg4(C1[i1/dim0][j1/dim0][i1%dim0][j1%dim0]);
        }

#else
   
      #pragma loop_coalesce 2
      for(uint i1=0; i1<dim1; ++i1)
      {
    
        #pragma unroll dim0 
        for(uint j1=0; j1<dim1; ++j1)
        {

          float volatile _pC[dim0] __attribute((register));
    
          #pragma unroll dim0 
          for(uchar _i=0; _i<dim0; ++_i)
            if(_i == i1%dim0)
              _pC[_i] = __fpga_reg2(C1[i1/dim0][j1/dim0][_i][j1%dim0]);
    
          const ulong iC = dim2*( (ulong)(I2*dim1) + i1 ) + (ulong)(J2*dim1) + j1;
          gC[iC]=__fpga_reg4(_pC[i1%dim0]);
    
        }
      }    

#endif

  }  // I2 J2

}

#define kern(NUM, I2BEGIN, I2END) \
__kernel void \
__attribute__((max_work_group_size(1,1,1))) \
__attribute__((uses_global_work_offset(0))) \
krnl_cannon_##NUM( __global volatile const float * restrict gA, \
                   __global volatile const float * restrict gB, \
                   __global volatile float * restrict gC, \
                   uint dim ) \
{ \
  cannon_func(dim, I2BEGIN, I2END, gA, gB, gC); \
} \

#define CHUNCK (((dim/dim1)+1)/2)

kern(0, 0, CHUNCK)
kern(1, CHUNCK, (dim/dim1))

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
