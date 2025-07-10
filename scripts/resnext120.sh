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
module load cuda/12.6.3
python --version

module load python3/3.12.9

source ../envs/cnn312/bin/activate

python --version

pip install --upgrade pip
pip install -r requirementscnn.txt


python main.py --model-name "resnext101_32x8d" --source-dir "/work3/s233780/CircularMaskedDataSorted/120min"