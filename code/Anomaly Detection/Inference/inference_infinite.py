# %% [markdown]
# # Imports

# %%
import sys, argparse, json
sys.path.append('..') #adjust based on your system's directory
import torch, time, os 
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import psutil, gc, os
import time, platform
from inference import get_cpu_info, get_ram_info

#Load Data
def load_data(data_path, device):
    return torch.load(data_path, map_location = device, weights_only=False)

#Load Model
def load_model(model_path, device):
    model = torch.load(model_path, map_location = device, weights_only=False)
    return model.module.eval()

#Use DataLoader for iterating over batches
def data_loader(data, batch_size):
    return DataLoader(data, batch_size = batch_size, drop_last = False)   #Drop samples out of the batch size

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_model_memory_mb(model):
    total_params = sum(p.numel() for p in model.parameters())
    param_bytes = total_params * 4  # assuming float32
    return param_bytes / 1024 / 1024

# %% [markdown]
# ### Size Calculator

# %%
import torch
from typing import Union

def get_tensor_size_in_mb(tensor: torch.Tensor) -> Union[float, str]:
    """
    Calculates the size of a PyTorch tensor in megabytes (MB).

    This function determines the total memory occupied by a tensor by multiplying
    the number of elements in the tensor by the size of a single element in bytes.
    The result is then converted from bytes to megabytes for a more
    human-readable format.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to be measured.

    Returns:
        Union[float, str]: The size of the tensor in MB as a float.
                         Returns a string message if the input is not a torch.Tensor.
    """
    # Check if the input is a PyTorch tensor to prevent errors.
    if not isinstance(tensor, torch.Tensor):
        return "Input must be a PyTorch tensor."

    try:
        # Step 1: Get the total number of elements in the tensor.
        # torch.numel() returns the number of elements in a tensor,
        # which is equivalent to the product of its dimensions.
        num_elements = tensor.numel()

        # Step 2: Get the size of a single element in bytes.
        # tensor.element_size() returns the size of each element
        # based on its data type (e.g., float32 is 4 bytes, float64 is 8 bytes).
        element_size_in_bytes = tensor.element_size()

        # Step 3: Calculate the total size in bytes.
        total_size_in_bytes = num_elements * element_size_in_bytes

        # Step 4: Convert the total size to megabytes.
        # There are 1024 bytes in a kilobyte and 1024 kilobytes in a megabyte.
        size_in_mb = total_size_in_bytes / (1024 * 1024)

        return size_in_mb
    except Exception as e:
        # A simple try-except block to catch potential runtime errors.
        return f"An error occurred: {e}"

# %% [markdown]
# ### InfiniteSampler

# %%
from torch.utils.data.sampler import Sampler
import distributed
from typing import Any, Optional
import itertools

def _get_torch_dtype(size: int) -> Any:
    return torch.int32 if size <= 2**31 else torch.int64

def _generate_randperm_indices(*, size: int, generator: torch.Generator):
    """Generate the indices of a random permutation."""
    dtype = _get_torch_dtype(size)
    # This is actually matching PyTorch's CPU implementation, see: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L900-L921
    perm = torch.arange(size, dtype=dtype)
    for i in range(size):
        j = torch.randint(i, size, size=(1,), generator=generator).item()

        # Always swap even if no-op
        value = perm[j].item()
        perm[j] = perm[i].item()
        perm[i] = value
        yield value

class InfiniteSampler(Sampler):
    def __init__(
        self,
        *,
        sample_count: int,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        step: Optional[int] = None,
        advance: int = 0,
    ):
        self._sample_count = sample_count
        self._seed = seed
        self._shuffle = shuffle
        self._start = distributed.get_global_rank() if start is None else start
        self._step = distributed.get_global_size() if step is None else step
        self._advance = advance

    def __iter__(self):
        if self._shuffle:
            iterator = self._shuffled_iterator()
        else:
            iterator = self._iterator()

        yield from itertools.islice(iterator, self._advance, None)

    def _iterator(self):
        assert not self._shuffle

        while True:
            iterable = range(self._sample_count)
            yield from itertools.islice(iterable, self._start, None, self._step)

    def _shuffled_iterator(self):
        assert self._shuffle

        # Instantiate a generator here (rather than in the ctor) to keep the class
        # picklable (requirement of mp.spawn)
        generator = torch.Generator().manual_seed(self._seed)

        while True:
            iterable = _generate_randperm_indices(size=self._sample_count, generator=generator)
            yield from itertools.islice(iterable, self._start, None, self._step)


def inference(
    model, data, real_redshift, device, 
    batch_size, num_workers=0,disable_progress=False
):
    # %%
    get_cpu_info()
    get_ram_info()
    # %%
    sample_count = len(data.tensors[0])
    sampler = InfiniteSampler(
        sample_count=sample_count,
        shuffle=False,
        seed=7,
        advance=0,
    )

    # result jsons will be saved in this folder
    if not os.path.exists(device):
        os.makedirs(device, exist_ok=True)

    # %%
    dataloader = DataLoader(
        data, batch_size = batch_size, drop_last = False,
        sampler=sampler, num_workers=num_workers
    )

    # %%
    total_size = sum([get_tensor_size_in_mb(t) for t in data.tensors])
    mb_per_sample = total_size / len(data.tensors[0])

    print(f'Total size {total_size:0.5f}MB, MB per sample {mb_per_sample:0.5f}')
    # one sample is ~ 0.019573 MB in size

    batch_per_GB = int(np.ceil(1024 / (batch_size * mb_per_sample)))
    print(f'Batches per GB {batch_per_GB}')

    # %%
    total_batches = batch_per_GB * 1024 # 1TB 
    total_time = 0.0  # Initialize total time for execution
    total_data_bits = 0  # Initialize total data bits processed

    start = time.perf_counter()
    total = []
    # Initialize the profiler to track both CPU and GPU activities and memory usage
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=total_batches, disable=disable_progress):
            image = data[0].to(device)  # Image to device
            magnitude = data[1].to(device)  # Magnitude to device

            _ = model([image, magnitude])  # Model inference

            # Calculate data size for this batch
            image_bits = image.element_size() * image.nelement() * 8  # Convert bytes to bits
            magnitude_bits = magnitude.element_size() * magnitude.nelement() * 8  # Convert bytes to bits
            total_data_bits += image_bits + magnitude_bits  # Add data bits for this batch

            if (i+1) % batch_per_GB !=0:
                if i+1 >= total_batches: break
                else: continue

            GB_processed = int((i+1) / batch_per_GB)
            print(f'Processed {GB_processed} GB.')

            # benchmark results
            num_samples = (i+1) * batch_size
            # Extract total CPU and GPU time
            total_time = time.perf_counter() - start 
            total_process_mem = get_process_memory_mb()
            execution_info = {
                'total_execution_time (seconds)': total_time,
                'total_process_memory (MB)': total_process_mem,
                'num_batches': i+1,   # Number of batches
                'batch_size': batch_size,   # Batch size
                'device': device,   # Selected device,
                'GB_processed': GB_processed
            }

            avg_time_batch = total_time / (i+1)

            # Average execution time per batch
            execution_info['execution_time_per_batch'] = avg_time_batch
            # Throughput in bits per second (using total_time for all batches)
            execution_info['throughput_bps'] = total_data_bits / total_time
            execution_info['sample_persec'] = num_samples / total_time,  # Number of samples processed per second
            
            total.append(execution_info)
            with open(f'{device}/{GB_processed}.json', 'w') as f:
                json.dump(execution_info, f, indent=4)

            # stop condition, otherwise infinite sampler will keep running forever
            if i+1 >= total_batches: break

    with open(f'{device}/total.json', 'w') as f:
        json.dump(total, f, indent=4)

def engine(args):
    data = load_data(args.data_path, args.device)
    model = load_model(args.model_path, args.device)
    inference(
        model, data, data[:][2], device = args.device, 
        batch_size = args.batch_size,num_workers=args.num_workers,
        disable_progress=args.disable_progress
    )

if __name__ == '__main__':
    prj_dir = '../' #adjust based on your system's directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--data_path', type = str, default = '../../../raw_data/1.pt')
    parser.add_argument('--model_path', type = str, default  = prj_dir + 'Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt')
    parser.add_argument('--device', type = str, default = 'cpu')    # To run on GPU, put cuda, and on CPU put cpu
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--disable_progress', action='store_true', default=False)

    args = parser.parse_args()
    engine(args)
