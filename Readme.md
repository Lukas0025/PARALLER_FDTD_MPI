# Finite-difference time-domain method (FDTD) MPI Implementation

## Thermal simulation akcelarated on Barbora (IT4I.cz) supercomputer

This repository contains implementation of MPI parallel FDTD on 2D plane with support for 1D and 2D decomposition. Implementing P2P and RMA mode and support hybrid mode with OMP. Output file can by save using seq method (Save on root rank) or par method using HDF5 library.

###### Author: Lukáš Plevač <xpleva07@vutbr.cz> 
###### Semetral project to PPP at BUT FIT

### Setup HPC

Load modules using ml.

```bash
#!/bin/bash

ml purge # purge loaded modules
ml CMake/3.22.1-GCCcore-11.2.0 intel/2021b HDF5/1.12.1-intel-2021b-parallel
ml Score-P/8.0-iimpi-2021b # pro profilovani
```

## Build project

Simply usimg make it make build dir and setup build using cmake

```bash
make
```

## Run project

```
Usage:
Mandatory arguments:
  -m [0-2]    mode 0 - run sequential version
              mode 1 - run parallel version point-to-point
              mode 2 - run parallel version RMA
  -n          number of iterations
  -i          material HDF5 file
Optional arguments:
  -t          number of OpenMP threads (default 1)
  -o          output HDF5 file
  -w          disk write intensity (every N-th step)
  -a          air flow rate (values in <0.0001, 0.5> make sense)
  -d          debug mode (copare results of SEQ and PAR versions and print result)
  -v          verification mode (copare results of SEQ and PAR versions)
  -p          parallel I/O mode
  -r          render results (with -d or -v) into *.png image.
  -g          Use 2D decomposition instead of 1D.
  -b          batch mode - output data into CSV format
  -h          batch mode - print CSV header
```

example run:

```
mpirun -np 16 ./build/ppp_proj01 -m 1 -n 78125 -i scripts/input_data_1024.h5 -d -r hello.png -g
```

## Setup git on Barbora

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/git
```
