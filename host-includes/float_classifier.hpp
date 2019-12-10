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

#ifndef __FLOAT_CLASSIFIER_HPP__
#define __FLOAT_CLASSIFIER_HPP__

template<typename FLOAT_TYPE>
class float_classifier
{

  private:

  size_t _infs, _nans, _normals, _subnormals, _zeros;
  size_t _unknowns;
  size_t _total;

  public:

  float_classifier() : 
    _infs(0), _nans(0), _normals(0), _subnormals(0), _zeros(0),
    _unknowns(0), _total(0) {};

  void eval(FLOAT_TYPE f)
  {
    switch(std::fpclassify(f))
    {
      case FP_INFINITE:  ++_infs;       break;       
      case FP_NAN:       ++_nans;       break;
      case FP_NORMAL:    ++_normals;    break;
      case FP_SUBNORMAL: ++_subnormals; break;
      case FP_ZERO:      ++_zeros;      break;
      default:           ++_unknowns;
    }
    ++_total;
  }

  size_t infs()       const { return _infs; }
  size_t nans()       const { return _nans; }
  size_t normals()    const { return _normals; }
  size_t subnormals() const { return _subnormals; }
  size_t zeros()      const { return _zeros; }
  size_t unknowns()   const { return _unknowns; }
  size_t total()      const { return _total; }

};

#endif

