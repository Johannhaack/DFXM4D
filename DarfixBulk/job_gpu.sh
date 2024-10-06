### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J DFXM
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- set the email address -- 
#BSUB -u johann.haack@gmail.com
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err

source /dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/Johann_Haack/miniconda3/bin/activate darfixgpu
module load cuda
python /zhome/a7/7/183900/Thesis/CellTracking/DarfixBulk/create_volume_gpu.py

### You can start the job with: bsub < jobscript.sh
### Get the job id via bjobs