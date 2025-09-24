import sys, argparse, json
sys.path.append('..') #adjust based on your system's directory
import torch, time
from tqdm import tqdm
import os, time
from inference import get_cpu_info, get_ram_info, load_data, load_model, data_loader, get_process_memory_mb

# Define the inference function with profiling for both CPU and GPU memory usage
def inference(
    model, dataloader, num_samples, 
    device, batch_size, disable_progress=False
):
    total_time = 0.0  # Initialize total time for execution
    num_batches = 0   # Initialize number of batches
    total_data_bits = 0  # Initialize total data bits processed

    start = time.perf_counter()
    # Initialize the profiler to track both CPU and GPU activities and memory usage
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader), total=len(dataloader), 
            disable=disable_progress
        ):
            image = data[0].to(device)  # Image to device
            magnitude = data[1].to(device)  # Magnitude to device

            _ = model([image, magnitude])  # Model inference

            # Append the redshift prediction to analysis list
            num_batches += 1

            # Calculate data size for this batch
            image_bits = image.element_size() * image.nelement() * 8  # Convert bytes to bits
            magnitude_bits = magnitude.element_size() * magnitude.nelement() * 8  # Convert bytes to bits
            total_data_bits += image_bits + magnitude_bits  # Add data bits for this batch
    
    # Extract total CPU and GPU time
    total_time = time.perf_counter() - start 
    total_process_mem = get_process_memory_mb()
    execution_info = {
            'total_execution_time (seconds)': total_time,
            'total_process_memory (MB)': total_process_mem,
            'num_batches': num_batches,   # Number of batches
            'batch_size': batch_size,   # Batch size
            'device': device,   # Selected device
        }
 
    avg_time_batch = total_time / num_batches

    # Average execution time per batch
    execution_info['execution_time_per_batch'] = avg_time_batch
    # Throughput in bits per second (using total_time for all batches)
    execution_info['throughput_bps'] = total_data_bits / total_time
    execution_info['sample_persec'] = num_samples / total_time,   # Number of samples processed per second
    return execution_info

#This is the engine module for invoking and calling various modules
def engine(args):
    # --- NEW: Workload distribution logic moved to Python ---
    total_files = args.total_files
    # Get task ID and total tasks directly from environment variables set by Slurm
    # This is more reliable than using argparse for these values
    try:
        task_id = int(os.environ['SLURM_PROCID'])
        total_tasks = int(os.environ['SLURM_NTASKS'])
    except KeyError:
        print("Warning: SLURM environment variables not found. Defaulting to serial run.")
        task_id = 0
        total_tasks = 1

    print(f"Starting task id {task_id} of {total_tasks}...")

    files_per_task = total_files // total_tasks
    remainder = total_files % total_tasks

    start_index = task_id * files_per_task
    if task_id < remainder:
        start_index += task_id
    else:
        start_index += remainder

    end_index = start_index + files_per_task
    if task_id < remainder:
        end_index += 1

    file_indices = range(start_index, end_index)
    # --- END of Workload distribution logic ---

    model = load_model(args.model_path, args.device)
    available_files = sorted(os.listdir(args.data_dir)) # Sort the file list to ensure consistent indexing
    N = len(available_files)
    
    all_job_stats = []
    for file_idx in file_indices:
        # Check if the file index is valid
        # if file_idx >= len(available_files):
        #     print(f"Warning: File index {file_idx} is out of bounds. Skipping.")
        #     continue
            
        filename = available_files[file_idx % N] # reuse files, so that we can extend the dataset beyond its original size limit
        data_path =  os.path.join(args.data_dir, filename)
        print(f"--- Processing file index: {file_idx} at path: {data_path} ---")
        
        # Check if file exists
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found at {data_path}. Skipping.")
            continue
            
        data = load_data(data_path, args.device)
        dataloader = data_loader(data, args.batch_size, num_workers=args.num_workers)
        
        file_stats = inference(
            model, dataloader, len(data[:][2]), device=args.device, 
            batch_size=args.batch_size,
            disable_progress=args.disable_progress
        )
        all_job_stats.append({
            "file_index": file_idx,
            "stats": file_stats
        })

    # Save aggregated results for this job to a unique file name
    output_dir = args.output_dir
    output_filename = os.path.join(output_dir, f'job_{task_id}.json')
    print(f"Saving aggregated results to {output_filename}")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(all_job_stats, f, indent=4)

    
# Pathes and other inference hyperparameters can be adjusted below
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    
    # NEW: Directory containing the .pt files
    parser.add_argument('--data_dir', type=str, default='../../../raw_data/100MB')
    
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output JSON file for this job.')

    parser.add_argument('--model_path', type = str, default  = '../Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt')
    parser.add_argument('--device', type = str, default = 'cpu', choices=['cpu', 'cuda'])     # To run on GPU, put cuda, and on CPU put cpu
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--disable_progress', action='store_true', default=False)
    
    # --- NEW: Arguments for concurrent jobs ---
    parser.add_argument('--total_files', type=int, required=True)

    args = parser.parse_args()

    # get_cpu_info()
    # get_ram_info()
    engine(args)
