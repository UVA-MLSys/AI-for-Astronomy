import subprocess
import argparse
import re
from datetime import datetime

def parse_slurm_time_to_seconds(time_str):
    """Converts Slurm's time format (D-HH:MM:SS or HH:MM:SS) to seconds."""
    parts = time_str.split('-')
    days = int(parts[0]) if len(parts) > 1 else 0
    time_parts = parts[-1].split(':')
    
    sec = 0
    if len(time_parts) == 3:
        sec = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
    elif len(time_parts) == 2:
        sec = int(time_parts[0]) * 60 + int(time_parts[1])
    elif len(time_parts) == 1:
        sec = int(time_parts[0])
        
    return days * 86400 + sec

def main(job_id):
    """Queries Slurm for job stats and prints a detailed summary."""
    print(f"Fetching stats for master job ID: {job_id}")
    
    # Get detailed stats for all tasks in the job array
    cmd = [
        'sacct', '-j', job_id,
        '--format=JobID,Start,End,State',
        '-p', '--noheader'
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error calling sacct: {e.stderr}")
        return

    lines = result.stdout.strip().split('\n')
    lines = [line for line in lines if '.batch' not in line]
    
    min_start_time = None
    max_end_time = None
    
    task_count = 0
    completed_jobs = 0
    OOM = 0
    
    for line in lines:
        # We only care about array tasks, which have a '_' in the JobID
        if not line or '_' not in line.split('|')[0]:
            continue
        
        task_count += 1
        print(line.strip().split('|'))
        jobid, start_str, end_str, state, _ = line.strip().split('|')

        # Slurm timestamp format is YYYY-MM-DDTHH:MM:SS
        # We only consider tasks that actually completed
        if state.strip() == 'COMPLETED':
            completed_jobs += 1
            current_start = datetime.fromisoformat(start_str)
            current_end = datetime.fromisoformat(end_str)

            if min_start_time is None or current_start < min_start_time:
                min_start_time = current_start
            
            if max_end_time is None or current_end > max_end_time:
                max_end_time = current_end
        elif state.strip() == 'OUT_OF_MEMORY':
            OOM += 1

    if task_count == 0 or min_start_time is None:
        print("No completed tasks found for this job array.")
        return
    if OOM > 0:
        print(f'Warning !! {OOM} Out of memory errors detected.')

    if completed_jobs != len(lines):
        print(f'{completed_jobs} among {len(lines)} jobs are completed.') # same job appears as 'job_no' and 'job_no.batch', hence //2

    # 1. Manually calculate the wall time based on your definition
    calculated_duration = max_end_time - min_start_time
    
    # 2. Get the master job's reported Elapsed time from sacct
    cmd_master = ['sacct', '-j', job_id, '--format=Elapsed', '-n']
    result_master = subprocess.run(cmd_master, capture_output=True, text=True, check=True)
    master_elapsed_str = result_master.stdout.strip().split('\n')[0]
    
    print("\n--- True Wall-Time Calculation ---")
    print(f"Number of Tasks Analyzed: {task_count}")
    print(f"Earliest Task Start Time:   {min_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Latest Task End Time:       {max_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("--------------------------------------------------")
    print(f"Calculated Wall Time (End-Start): {calculated_duration}")
    print(f"Master Job 'Elapsed' Time (sacct):  00:{master_elapsed_str}") # Add leading zeros for consistency
    print("--------------------------------------------------")
    print("\nThese two values should match, confirming that the master job's 'Elapsed'")
    print("time correctly represents the total wall time of the array.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the true wall time for a Slurm job array.")
    parser.add_argument("job_id", help="The master Slurm job ID.")
    args = parser.parse_args()
    main(args.job_id)