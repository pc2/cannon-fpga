# cannon-fpga

This repository contains the codes related to the paper

Paolo Gorlani, Tobias Kenter, Christian Plessl, *OpenCL implementation of Cannon's matrix multiplication algorithm on Intel Stratix 10 FPGAs*, International Conference on Field-Programmable Technology, Tianjin 2019.

### Contents

 - `paper-kernels` contains the kernel codes reported in the paper.
 - `followup-kernels` contains the latest version of the kernel codes.
 - `host-src` and `host-includes` contain the host codes.
 

**NOTE:**: set `PLATFORM_ID` and `DEVICE_ID` in `host-src/host.cpp` in order to target your FPGA accelerator. 
