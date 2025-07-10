#!/bin/sh
### --- specify queue ---

#BSUB -q gpua100                            
#BSUB -J yolo_train                    
#BSUB -n 4                                 
#BSUB -gpu "num=1:mode=exclusive_process"   
#BSUB -W 24:00                               
#BSUB -R "rusage[mem=10GB]"                  
#BSUB -B                                    
#BSUB -N                                    
#BSUB -o yoloout_%J.out               # Output log
#BSUB -e yoloerr_%J.err               # Error log

# Load the CUDA module
module load cuda/11.8

source ../envs/pytorch/bin/activate


pip install --upgrade pip
pip install -r requirements.txt

python yolo_trainer.py