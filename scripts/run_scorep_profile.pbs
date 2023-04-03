#!/bin/bash
#PBS -A DD-22-68
#PBS -N PPP_PROJ01_PROF
#PBS -q qprod
#PBS -l walltime=01:00:00
#PBS -l select=4:ncpus=36:mpiprocs=4:ompthreads=6

ml CMake/3.22.1-GCCcore-11.2.0 intel/2021b HDF5/1.12.1-intel-2021b-parallel Score-P/8.0-iimpi-2021b
# ml CMake intel/2020a HDF5/1.10.6-intel-2020a-parallel Score-P/6.0-intel-2020a

cd "$PBS_O_WORKDIR"

STDOUT_FILE="run_scorep_profile_out.csv"
STDERR_FILE="run_scorep_profile_err.txt"
BINARY_PATH="../build_prof/ppp_proj01"

echo "" > $STDOUT_FILE

DISK_WRITE_INTENSITY=50

export SCOREP_ENABLE_PROFILING=true
export SCOREP_TOTAL_MEMORY=2G
mpirun -np 16 $BINARY_PATH -b -h -n 100 -t 6 -m 1 -w $DISK_WRITE_INTENSITY -i input_data_1024.h5 >> $STDOUT_FILE 2>> $STDERR_FILE
mpirun -np 16 $BINARY_PATH -b -g -n 100 -t 6 -m 1 -w $DISK_WRITE_INTENSITY -i input_data_1024.h5 >> $STDOUT_FILE 2>> $STDERR_FILE

export SCOREP_ENABLE_TRACING=true
export SCOREP_FILTERING_FILE=ppp_scorep_filter.flt
mpirun -np 16 $BINARY_PATH -b -h -n 100 -t 6 -m 1 -w $DISK_WRITE_INTENSITY -i input_data_1024.h5 >> $STDOUT_FILE 2>> $STDERR_FILE
mpirun -np 16 $BINARY_PATH -b -g -n 100 -t 6 -m 1 -w $DISK_WRITE_INTENSITY -i input_data_1024.h5 >> $STDOUT_FILE 2>> $STDERR_FILE
