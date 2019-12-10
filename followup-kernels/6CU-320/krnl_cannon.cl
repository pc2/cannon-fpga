#define dim0 8 
#define dim1 360 
#define NB1 (dim1/dim0)

#define __fpga_reg2(x) __fpga_reg(__fpga_reg(x)) 
#define __fpga_reg4(x) __fpga_reg(__fpga_reg(__fpga_reg(__fpga_reg(x))))

#define SHIFT 19 

inline void
cannon_func( const uint dim2, const ushort I2begin, const ushort I2end, 
             __global volatile const float * restrict gA, 
             __global volatile const float * restrict gB, 
             __global volatile float * restrict gC )
{

  const uint NB2 = dim2/dim1;
  const uint tot_it = (I2end-I2begin)*NB2*NB2 + 1;

  float __attribute((numbanks(dim0*dim0),max_concurrency(1), max_replicates(1))) C1[NB1][NB1][dim0][dim0];

  ushort I2 = I2begin;
  ushort J2 = 0;
  ushort K2 = 0;

  #pragma max_concurrency 1
  for(uint IDX=0; IDX<tot_it; ++IDX)
  {
  
    float A1[NB1][NB1][dim0][dim0] __attribute((numbanks(dim0*dim0)));
    float B1[NB1][NB1][dim0][dim0] __attribute((numbanks(dim0*dim0)));

    {
      const ushort pI2 = (J2 ? I2 : (I2-1));
      const ushort pJ2 = (J2 ? J2 : NB2) - 1;

      #pragma loop_coalesce 2
      for(ushort i1=0; i1<dim1; ++i1)
        #pragma unroll dim0 
        for(ushort j1=0; j1<dim1; ++j1)
        {
          const ulong iA = dim1*( (ulong)(I2*dim2) + NB2*i1 + K2 ) + j1;
          const ulong iB = dim1*( (ulong)(K2*dim2) + NB2*i1 + J2 ) + j1;

          A1[i1/dim0][j1/dim0][i1%dim0][j1%dim0] = (I2<I2end) ? (gA[iA]) : 0.0f;
          B1[j1/dim0][i1/dim0][i1%dim0][j1%dim0] = (I2<I2end) ? (gB[iB]) : 0.0f;
  
          const ulong iC = dim1*( (ulong)(pI2*dim2) + NB2*i1 + pJ2 ) + j1;
          if((!K2)&&IDX) gC[iC]=(C1[i1/dim0][j1/dim0][i1%dim0][j1%dim0]);
        } 
    }

    float __attribute((register)) C0[SHIFT][dim0][dim0];

    #pragma loop_coalesce 3 
    #pragma ivdep array(C1)
    for(ushort W=0; W<NB1*NB1; W+=SHIFT)
      for(uchar K1=0; K1<NB1; ++K1)
        #pragma ivdep array(C1)
        for(uchar S=0; S<SHIFT; ++S)
        {

          const bool valid = (W+S)<NB1*NB1;  
          const uchar I1 = (W+S)/NB1;  
          const uchar J1 = (W+S)%NB1;

          float A0[dim0][dim0+1];
          float B0[dim0+1][dim0];

          // READ A AND B 
          #pragma unroll dim0
          for(uchar i=0; i<dim0; ++i)
            #pragma unroll dim0
            for(uchar j=0; j<dim0; ++j)
            {

              A0[i][(j+dim0-i)%dim0] = (valid) ? A1[I1][K1][i][j] : 0.0f;
              B0[(i+dim0-j)%dim0][j] = (valid) ? B1[J1][K1][i][j] : 0.0f;  // reversed index
              const float _t = ((!K1)&&valid) ? ( (K2) ? C1[I1][J1][i][j] : 0.0f ) : C0[0][i][j];
              #pragma unroll 
              for(uchar _s=0; _s<SHIFT-1; ++_s)
                C0[_s][i][j] =  C0[_s+1][i][j];
              C0[SHIFT-1][i][j] = _t;

            }
 
          // COMPUTE 
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
                C0[SHIFT-1][i][j] += __fpga_reg(A0[i][j]) * __fpga_reg(B0[i][j]);
                A0[i][j] = __fpga_reg(A0[i][j+1]); 
                B0[i][j] = __fpga_reg(B0[i+1][j]);
              }
                
          }

          // WRITE RESULT
          if((K1==NB1-1) && valid)
            #pragma unroll dim0
            for(uchar i=0; i<dim0; ++i)
              #pragma unroll dim0
              for(uchar j=0; j<dim0; ++j)
              {    
                C1[I1][J1][i][j] = C0[SHIFT-1][i][j];
              }

        } // W K1 S 

    K2 = __fpga_reg((K2!=NB2-1) ? K2+1 : 0); 
    J2 = __fpga_reg((K2) ? J2 : ((J2!=NB2-1) ? J2+1 : 0));
    I2 = __fpga_reg((J2||K2) ? I2 : I2+1);

  } // K2 I2 J2

}

#define kern(NUM) \
__kernel void \
__attribute__((max_work_group_size(1,1,1))) \
__attribute__((uses_global_work_offset(0))) \
krnl_cannon_##NUM( __global volatile const float * restrict gA, \
                   __global volatile const float * restrict gB, \
                   __global volatile float * restrict gC, \
                   uint dim, uint I2begin, uint I2end ) \
{ \
  cannon_func(dim, I2begin, I2end, gA, gB, gC); \
} \

kern(0)
kern(1)
kern(2)
kern(3)
kern(4)
kern(5)


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
