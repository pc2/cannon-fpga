## Kernel codes

 |  directory | # Compute Units  | Block size (`dim1`) |
 |:----------:|:----------------:|:-------------------:|
 | `1CU-1024` |        1         |         1024        |
 | `2CU-720`  |        2         |         720         |
 | `3CU-512`  |        3         |         512         |
 | `4CU-512`  |        4         |         512         |
 | `5CU-512`  |        5         |         512         |
 | `6CU-360`  |        6         |         360         |

Each directory contains a `.cl` file and the `acl_quartus_report.txt` files related to the syntheses with different seeds.

## Synthesis command 

In order to synthesize a kernel, enter a directory (e.g. `5CU-512`) and run the following command 

```
$ aoc -v -v -v -report -g -W -high-effort -board=p520_max_sg280l -fp-relaxed -fpc -no-interleaving=default -global-ring -duplicate-ring -seed=<seed> krnl_cannon.cl -o krnl_cannon-s<seed>.aocx
```

The kernel codes reported in the paper are synthesized with Intel FPGA SDK for OpenCL 19.1 with the `p520_max_sg280l` BittaWare 520N Board Support Package based on Quartus 19.1.

## Host code

The host code runs the kernels and tests the FPGA solution against the one computed on the host. 

The code can be compiled as follow
```
 $ make host
```

The generated `host` executable must be called with the following parameters
```
 $ ./host <#CU> <dim> <.aocx file path>
```

Where
 - `<#CU>` is the number of compute units (i.e. the number of compute units in the .aocx file),
 - `<dim>` is the size of the test matrices, must be a multiple of the block size (`dim1`),
 - `<.aocx file path>` is the path to the `.aocx` file.

**Note 1:** `<#CU>` must match with the number compute units of the `.aocx` file.

