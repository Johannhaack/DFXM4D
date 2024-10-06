#!/bin/sh 
### General options 
### -- set the job Name -- 
#BSUB -J DFXM
### -- ask for number of cores (default: 1) -- 
#BSUB -n 8  # Request 16 cores
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=36GB]"  # Request 32GB per process
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
###BSUB -M 40GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 6:00
### -- set the email address -- 
#BSUB -u johann.haack@gmail.com
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/Output_%J.out 
#BSUB -e logs/Output_%J.err

source /dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Johann_Haack/miniconda3/bin/activate darfixbase

python /zhome/a7/7/183900/Thesis/CellTracking/DarfixBulk/multiprocess_mosaicityplot.py

### You can start the job with: bsub < jobscript.sh
### Get the job id via bjobs