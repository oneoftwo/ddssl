#!/bin/bash
#PBS -N LJW_1
#PBS -l nodes=gnode5:ppn=4
#PBS -l walltime=1000:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

source activate ljw 
cd ..
python -u train.py 1>output_1.txt
