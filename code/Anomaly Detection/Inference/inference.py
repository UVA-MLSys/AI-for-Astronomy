import sys, argparse, json
sys.path.append('..') #adjust based on your system's directory
import torch, time, os 
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil, gc
import time, platform

def get_cpu_info():
    # CPU Information
    print("CPU Information:")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"System: {platform.system()}")
    print(f"Platform: {platform.platform()}")

    return {
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'system': platform.system(),
        'platform': platform.platform()
    }

# RAM Information
def get_ram_info():
    if hasattr(os, 'sysconf'):
        if 'SC_PAGE_SIZE' in os.sysconf_names and 'SC_PHYS_PAGES' in os.sysconf_names:
            page_size = os.sysconf('SC_PAGE_SIZE')  # in bytes
            total_pages = os.sysconf('SC_PHYS_PAGES')
            total_ram = page_size * total_pages  # in bytes
            total_ram_gb = total_ram / (1024 ** 3)  # convert to GB
            print(f"Total memory (GB): {total_ram_gb:.2f}")
            return total_ram_gb
    return None

#Load Data
def load_data(data_path, device):
    return torch.load(data_path, map_location = device, weights_only=False)

#Load Model
def load_model(model_path, device):
    model = torch.load(model_path, map_location = device, weights_only=False)
    return model.module.eval()

#Use DataLoader for iterating over batches
def data_loader(data, batch_size, num_workers=None):
    return DataLoader(
        data, batch_size = batch_size, drop_last = False,
        num_workers=num_workers
    )   #Drop samples out of the batch size

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_model_memory_mb(model):
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = total_params * 4  # assuming float32
    return param_bytes / 1024 / 1024

# Define the inference function with profiling for both CPU and GPU memory usage
def inference(model, dataloader, num_samples, device, batch_size):
    total_time = 0.0  # Initialize total time for execution
    num_batches = 0   # Initialize number of batches
    total_data_bits = 0  # Initialize total data bits processed

    start = time.perf_counter()
    # Initialize the profiler to track both CPU and GPU activities and memory usage
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
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
    execution_info['sample_persec'] = num_samples / total_time,  # Number of samples processed per second
    print(execution_info)

#This is the engine module for invoking and calling various modules
def engine(args):
    data = load_data(args.data_path, args.device)
    dataloader = data_loader(data, args.batch_size)
    model = load_model(args.model_path, args.device)
    inference(
        model, dataloader, len(data[:][2]), device = args.device, 
        batch_size = args.batch_size
    )

    
# Pathes and other inference hyperparameters can be adjusted below
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type = str, default = 'resized_inference.pt')
    parser.add_argument('--model_path', type = str, default  = '../Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt')
    parser.add_argument('--device', type = str, default = 'cuda', choices=['cpu', 'cuda'])    # To run on GPU, put cuda, and on CPU put cpu

    args = parser.parse_args()

    get_cpu_info()
    get_ram_info()
    engine(args)