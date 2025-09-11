#!/bin/bash

#SBATCH --job-name=inference_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1   # Each job runs on a single core
#SBATCH --mem=16G            # Adjust memory as needed
#SBATCH --time=01:00:00       # Adjust time limit as needed
#SBATCH --account=mlsys
#SBATCH --partition=standard # Adjust partition as needed, e.g., gpu, standard, etc.

# --- Output/Error logging ---
# Create directories for logs and results if they don't exist
mkdir -p slurm_logs
mkdir -p results

# %A is the master job ID, %a is the array task ID
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --error=slurm_logs/job_%A_%a.err

# --- Job Array Configuration ---
# This will be set via the command line for flexibility

# --- SCRIPT ARGUMENTS (passed from sbatch command) ---
# $1: Total number of files to process
# $2: Number of files each job should handle
TOTAL_FILES=$1
FILES_PER_JOB=$2

# --- Environment Setup ---
# Activate your conda/virtual environment
# source /path/to/your/miniconda3/bin/activate your_env_name

echo "Starting job array task ${SLURM_ARRAY_TASK_ID}"
echo "Total files: ${TOTAL_FILES}"
echo "Files per job: ${FILES_PER_JOB}"

# --- Calculate file indices for this specific job ---
# This math distributes the files among the array tasks
START_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) * FILES_PER_JOB + 1 ))
END_INDEX=$(( SLURM_ARRAY_TASK_ID * FILES_PER_JOB ))

# Ensure END_INDEX does not exceed TOTAL_FILES
if [ "$END_INDEX" -gt "$TOTAL_FILES" ]; then
  END_INDEX=$TOTAL_FILES
fi

# Generate the list of file indices for the python script
FILE_INDICES=$(seq $START_INDEX $END_INDEX)

echo "This task will process file indices: ${FILE_INDICES}"

# --- Check if there is anything to process ---
if [ "$START_INDEX" -gt "$TOTAL_FILES" ]; then
  echo "No files to process for this task. Exiting."
  exit 0
fi

# --- Execute the Python script ---
python inference_parallel.py \
    --batch_size 512 \
    --data_dir '../../../raw_data/' \
    --model_path '../Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt' \
    --device 'cpu' \
    --disable_progress \
    --file_indices ${FILE_INDICES} \
    --output_file "results/job_${SLURM_ARRAY_TASK_ID}.json"

echo "Job array task ${SLURM_ARRAY_TASK_ID} finished."