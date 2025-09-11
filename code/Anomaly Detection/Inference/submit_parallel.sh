#!/bin/bash

#SBATCH --job-name=inference_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --account=mlsys
#SBATCH --partition=standard # Adjust partition as needed, e.g., gpu, standard, etc.

mkdir -p outputs
mkdir -p results

#SBATCH --output=outputs/cpu_%A_%a.out

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

# --- NEW: Define output filename as a variable ---
OUTPUT_JSON_FILE="results/job_${SLURM_ARRAY_TASK_ID}.json"

# --- Execute the Python script ---
python inference_parallel.py \
    --batch_size 512 \
    --data_dir '../../../raw_data/100MB' \
    --model_path '../Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt' \
    --device 'cpu' \
    --disable_progress \
    --file_indices ${FILE_INDICES} \
    --output_file "${OUTPUT_JSON_FILE}"

echo "Python script finished. Now capturing Slurm stats."

# --- NEW SECTION: Capture Slurm stats and append to JSON ---

# Allow a few seconds for accounting data to be written
sleep 5 

# Get job stats using sacct. The format is JobID, Elapsed time, Max Memory
# The job step ID is $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID for array jobs
STATS_LINE=$(sacct -j $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID --format=JobID,Elapsed,MaxRSS,TotalCPU -n -p)

if [ -n "$STATS_LINE" ]; then
    # Parse the line (it's pipe-delimited because of the -p flag)
    IFS='|' read -r jobid elapsed maxrss totalcpu <<< "$STATS_LINE"
    
    echo "Slurm Stats Found: Elapsed=${elapsed}, MaxRSS=${maxrss}, TotalCPU=${totalcpu}"
    
    # Use jq to add the slurm stats to the JSON file.
    # This command creates a temporary file and then replaces the original.
    jq --arg elapsed "$elapsed" --arg maxrss "$maxrss" --arg totalcpu "$totalcpu" \
    '. + {slurm_stats: {elapsed_time: $elapsed, max_memory: $maxrss, total_cpu_time: $totalcpu}}' \
    "${OUTPUT_JSON_FILE}" > "${OUTPUT_JSON_FILE}.tmp" && mv "${OUTPUT_JSON_FILE}.tmp" "${OUTPUT_JSON_FILE}"

    echo "Successfully appended stats to ${OUTPUT_JSON_FILE}"
else
    echo "Warning: Could not retrieve Slurm stats for job ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
fi

echo "Job array task ${SLURM_ARRAY_TASK_ID} finished."

## Sample usage

# sbatch --array=<start>-<end> <script> <total_files> <files_per_job> 
# sbatch --array=1-1 submit.sbatch 10 10