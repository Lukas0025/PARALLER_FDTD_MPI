# Finite-difference time-domain method (FDTD) MPI Implementation

## Thermal simulation akcelarated on Barbora (IT4I.cz) supercomputer

### Author: Lukáš Plevač <xpleva07@vutbr.cz>

### Semetral project to PPP at BUT FIT

### Setup barbora

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

## Setup git on Barbora

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/git
```
