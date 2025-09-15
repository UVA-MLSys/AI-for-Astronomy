#!/bin/bash

#SBATCH --job-name=inference_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:10:00
#SBATCH --account=mlsys
#SBATCH --partition=standard # Adjust partition as needed, e.g., gpu, standard, etc.
#SBATCH --output=outputs/cpu_%A_%a.out
#---SBATCH --mail-type=begin,end
#---SBATCH --mail-user=mi3se@virginia.edu

# Load modules
# module load gcc nccl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p outputs
mkdir -p results

TOTAL_FILES=$1
FILES_PER_JOB=$2

# --- Environment Setup ---
source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate cosmic_ai

echo "Starting job array task ${SLURM_ARRAY_TASK_ID}"

START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * FILES_PER_JOB + 1 ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * FILES_PER_JOB ))

if [ "$END_INDEX" -gt "$TOTAL_FILES" ]; then
  END_INDEX=$TOTAL_FILES
fi

if [ "$START_INDEX" -gt "$TOTAL_FILES" ]; then
  echo "No files to process for this task. Exiting."
  exit 0
fi

FILE_INDICES=$(seq $START_INDEX $END_INDEX)
echo "This task will process file indices: ${FILE_INDICES}"

NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 1))

# It's good practice to ensure num_workers isn't negative if cpus-per-task is 1.
if [ "$NUM_WORKERS" -lt 0 ]; then
  NUM_WORKERS=0
fi
echo "Using ${NUM_WORKERS} workers for the PyTorch DataLoader."

# --- NEW: Define output filename as a variable ---
OUTPUT_JSON_FILE="results/1024GB_96/1/job_${SLURM_ARRAY_TASK_ID}.json"

# --- Execute the Python script ---
python inference_parallel.py \
    --batch_size 512 \
    --data_dir '../../../raw_data/100MB' \
    --model_path '../Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt' \
    --device 'cpu' \
    --disable_progress \
    --file_indices ${FILE_INDICES} \
    --output_file "${OUTPUT_JSON_FILE}" \
    --num_workers ${NUM_WORKERS} \
    --disable_progress

echo "Job array task ${SLURM_ARRAY_TASK_ID} finished."

## Sample usage

# sbatch --array=<start>-<end> <script> <total_files> <files_per_job> 
# sbatch --array=1-1 submit_parallel.sh 10 10
# sbatch --array=1-10 submit_parallel.sh 10 1

# 10GB
# sbatch --array=1-103 submit_parallel.sh 103 1

# 100 GB, note that Rivanna will max run 96 cores for a user at a time
# sbatch --array=1-1024 submit_parallel.sh 1024 1

# 1024 GB or 1TB data with 96 cores in parallel. So each gets 106.7 files
# sbatch --array=1-96 submit_parallel.sh 10240 107