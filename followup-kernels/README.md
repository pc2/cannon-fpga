### Kernel code

 | directory | # Compute Units  | Block size (`dim1`) |
 |:---------:|:----------------:|:-------------------:|
 | `5CU-512` |        5         |         512         |
 | `6CU-360` |        6         |         360         |
 | `7CU-360` |        7         |         360         |

Each directory contains a `.cl` file and the `acl_quartus_report.txt` files related to the syntheses with different seeds.

## Synthesis command 

In order to synthesize a kernel, enter a directory (e.g. `5CU-512`) and run the following command 

```
$ aoc -v -v -v -report -g -W -high-effort -board=p520_max_sg280l -fp-relaxed -fpc -no-interleaving=default -global-ring -duplicate-ring -seed=<seed> krnl_cannon.cl -o krnl_cannon-s<seed>.aocx
```

The kernel codes reported in the paper are synthesized with Intel FPGA SDK for OpenCL 19.3 with the `p520_hpc_sg280l` BittaWare 520N Board Support Package based on Quartus 19.2.


## Host code

The host code runs the kernels and tests the FPGA solution against the one computed on the host. 

Compile the code using make as follow
```
 $ make host
```

The generated `host` executable must be called with the following parameters
```
 host <#CU> <block-dim> <dim> <.aocx file path>
```

Where
 - `<#CU>` is the number of compute units (i.e. the number of kernels in the .aocx file)
 - `<block-dim>` is the block size of the synthesized code (i.e. `dim1` in the kernel code),
 - `<dim>` is size of the test matrices, must be a multiple of `block-dim`,
 - `<.aocx file path>` is the path to the `.aocx` file.

