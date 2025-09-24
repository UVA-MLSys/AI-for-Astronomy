#!/bin/bash

#SBATCH --job-name=inference_concurrent
#SBATCH --nodes=2 # min 2 for parallel partition
#SBATCH --ntasks=4 # min 4 for parallel partition
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=parallel
#SBATCH --account=mlsys
#SBATCH --mem-per-cpu=3G # the smallest memory that worked at 100MB partition
#SBATCH --output=outputs/concurrent_%j.out

# Load modules
# module load gcc nccl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p outputs
mkdir -p results

# --- Environment Setup ---
source /etc/profile.d/modules.sh
source ~/.bashrc

module load miniforge
conda activate cosmic_ai

echo "Starting job at $(date) with ${SLURM_NTASKS} tasks."
echo "Running on host: $(hostname)"
echo "This job is using ${SLURM_NNODES} nodes with ${SLURM_NTASKS} tasks."

DATA_SIZE_GB=$1 # first param is GB to data to be processed
PARTITION_SIZE_MB=100 # 100MB per file
TOTAL_FILES=$(echo "scale=0; ($DATA_SIZE_GB*1024+$PARTITION_SIZE_MB-1)/$PARTITION_SIZE_MB" | bc)
OUTPUT_DIR="results2/${DATA_SIZE_GB}GB/${2}"

echo -e "Total files: ${TOTAL_FILES}\n"

# --- Execute the Python script ---
srun python inference_parallel2.py \
    --batch_size 512 \
    --data_dir '../../../raw_data/100MB' \
    --model_path '../Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt' \
    --device 'cpu' \
    --disable_progress \
    --total_files ${TOTAL_FILES} \
    --output_dir ${OUTPUT_DIR} \
    --num_workers ${OMP_NUM_THREADS} \
    --disable_progress
    
# --- End Timer and Calculate Duration ---
DURATION=$SECONDS
echo -e "\nJob finished on: $(date)"
echo "Total duration: ${DURATION} seconds"

## Sample usage:
# This is a single job submission, not an array.
# sbatch submit_parallel.sh DATA_SIZE_GB RUN

# for 1GB and first run, highest 6000/64 tasks per node
# sbatch submit_parallel2.sh 1 1

# 10GB
# sbatch --nodes 2 --ntasks 103 submit_parallel2.sh 10 1

# 100GB 
# sbatch --nodes 24 --ntasks 1024 submit_parallel2.sh 100 1

# 256GB
# sbatch --nodes 48 --ntasks 2621 submit_parallel2.sh 256 1

# 512GB 
# sbatch --nodes 64 --ntasks 5242 submit_parallel2.sh 512 1

# 768GB
# sbatch --nodes 64 --ntasks 6000 submit_parallel2.sh 768 1 

# 1024GB or 1TB
# sbatch --nodes 64 --ntasks 6000 submit_parallel2.sh 1024 1