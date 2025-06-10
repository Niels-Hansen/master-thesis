#!/bin/sh
### --- specify queue ---

#BSUB -q gpua100                            
#BSUB -J resnext_train                    
#BSUB -n 4                                 
#BSUB -gpu "num=1:mode=exclusive_process"   
#BSUB -W 24:00                               
#BSUB -R "rusage[mem=10GB]"                  
#BSUB -B                                    
#BSUB -N                                    
#BSUB -o resnext_%J.out               # Output log
#BSUB -e resnext_%J.err               # Error log

# Load the CUDA module
module load cuda/11.8

source ../envs/pytorch/bin/activate

pip install --upgrade pip
pip install -r requirements.txt


python main.py --model resnext101_32x8d