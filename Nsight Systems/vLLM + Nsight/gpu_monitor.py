"""
GPU Memory Monitoring Utility using nvidia-ml-py (NVIDIA Management Library)

This script provides real-time GPU memory usage tracking during inference.
Use this to monitor GPU utilization while running vLLM inference.

Note: This tracks TOTAL GPU memory usage, which includes:
- Model weights (parameters)
- KV cache blocks
- Activations and temporary buffers
- CUDA graphs memory

Usage:
    python gpu_monitor.py

Requirements:
    pip install nvidia-ml-py
"""

import pynvml  # nvidia-ml-py provides pynvml module
import time
import os
from datetime import datetime

def initialize_nvml():
    """Initialize NVIDIA Management Library"""
    try:
        # initialize the nvidia management library to access gpu information
        pynvml.nvmlInit()
        print("✓ NVML initialized successfully")
        return True
    except pynvml.NVMLError as e:
        # if initialization fails, print error and return false
        print(f"✗ Failed to initialize NVML: {e}")
        return False

def get_gpu_info(device_index=0):
    """
    Get GPU device information
    
    Args:
        device_index: GPU device index (default: 0 for first GPU)
    
    Returns:
        dict: GPU information including name, total memory, driver version
    """
    try:
        # get a handle to the gpu device at the specified index (0 = first gpu)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        
        # query the gpu name (e.g., "NVIDIA GeForce GTX 1650")
        name = pynvml.nvmlDeviceGetName(handle)
        
        # get memory information object from the gpu
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # convert total memory from bytes to megabytes (divide by 1024^2)
        total_memory_mb = memory_info.total / (1024 ** 2)
        
        # get the nvidia driver version installed on the system
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        
        # return all gpu info as a dictionary, including the handle for later use
        return {
            "name": name,
            "total_memory_mb": total_memory_mb,
            "driver_version": driver_version,
            "handle": handle
        }
    except pynvml.NVMLError as e:
        # if any error occurs, print it and return none
        print(f"✗ Error getting GPU info: {e}")
        return None

def get_gpu_memory_usage(handle):
    """
    Get current GPU memory usage
    
    Args:
        handle: NVML device handle
    
    Returns:
        dict: Memory usage statistics (used, free, total in MB and %)
    """
    try:
        # query current memory state from the gpu using the handle
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # convert memory values from bytes to megabytes for readability
        total_mb = memory_info.total / (1024 ** 2)  # total vram on gpu
        used_mb = memory_info.used / (1024 ** 2)    # currently allocated vram
        free_mb = memory_info.free / (1024 ** 2)    # available vram
        
        # calculate what percentage of total memory is currently used
        utilization_percent = (used_mb / total_mb) * 100
        
        # return all memory stats rounded to 2 decimal places
        return {
            "total_mb": round(total_mb, 2),
            "used_mb": round(used_mb, 2),
            "free_mb": round(free_mb, 2),
            "utilization_percent": round(utilization_percent, 2)
        }
    except pynvml.NVMLError as e:
        # if query fails, print error and return none
        print(f"✗ Error getting memory usage: {e}")
        return None

def get_gpu_utilization(handle):
    """
    Get GPU compute utilization percentage
    
    Args:
        handle: NVML device handle
    
    Returns:
        int: GPU utilization percentage (0-100)
    """
    try:
        # query the gpu utilization rates (how busy the gpu cores are)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # return just the gpu core utilization as a percentage (0-100)
        # this shows how much the gpu is actively computing, not memory usage
        return utilization.gpu
    except pynvml.NVMLError as e:
        # if query fails, print error and return none
        print(f"✗ Error getting GPU utilization: {e}")
        return None

def monitor_gpu_continuous(device_index=0, interval=1.0, duration=None, log_file=None):
    """
    Continuously monitor GPU usage
    
    Args:
        device_index: GPU device index (default: 0)
        interval: Sampling interval in seconds (default: 1.0)
        duration: Total monitoring duration in seconds (None = infinite)
        log_file: Path to log file (None = print to console)
    """
    # step 1: initialize nvml library to communicate with nvidia driver
    if not initialize_nvml():
        return
    
    # step 2: get basic gpu information (name, memory, driver version)
    gpu_info = get_gpu_info(device_index)
    if not gpu_info:
        # if we can't get gpu info, exit the function
        return
    
    # step 3: open log file if specified, otherwise use console
    if log_file:
        # create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handle = open(log_file, 'w')
        output = file_handle
    else:
        output = None  # will use print()
    
    def log_message(msg):
        """helper function to write to both file and console"""
        # always print to console
        print(msg)
        # also write to file if log file is specified
        if output:
            output.write(msg + '\n')
            output.flush()  # ensure immediate write to file
    
    # print header with gpu information
    log_message(f"\n{'='*70}")
    log_message(f"GPU Monitor - {gpu_info['name']}")
    log_message(f"Total Memory: {gpu_info['total_memory_mb']:.2f} MB")
    log_message(f"Driver Version: {gpu_info['driver_version']}")
    log_message(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"{'='*70}\n")
    
    # step 4: extract the gpu handle for querying metrics
    handle = gpu_info['handle']
    # record when monitoring started (for duration tracking)
    start_time = time.time()
    
    try:
        # step 5: start infinite monitoring loop (until ctrl+c or duration limit)
        while True:
            # get current time for this measurement
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # query current memory usage from gpu
            mem_usage = get_gpu_memory_usage(handle)
            
            # query current gpu core utilization percentage
            gpu_util = get_gpu_utilization(handle)
            
            # if both queries succeeded, log the metrics in one line
            if mem_usage and gpu_util is not None:
                log_message(f"[{timestamp}] "
                           f"Memory: {mem_usage['used_mb']:.2f}/{mem_usage['total_mb']:.2f} MB "
                           f"({mem_usage['utilization_percent']:.2f}%) | "
                           f"GPU Util: {gpu_util}%")
            
            # check if we've been monitoring for the specified duration
            if duration and (time.time() - start_time) >= duration:
                break  # exit the loop if duration limit reached
            
            # wait for the specified interval before next measurement
            time.sleep(interval)
            
    except KeyboardInterrupt:
        # user pressed ctrl+c to stop monitoring
        log_message("\n\n✓ Monitoring stopped by user")
    finally:
        # always clean up: shutdown nvml library connection
        pynvml.nvmlShutdown()
        log_message("✓ NVML shutdown complete")
        # close log file if it was opened
        if output:
            output.close()

def get_snapshot(device_index=0):
    """
    Get a single snapshot of GPU usage
    
    Args:
        device_index: GPU device index (default: 0)
    
    Returns:
        dict: GPU usage snapshot
    """
    # initialize nvml library
    if not initialize_nvml():
        return None
    
    # get gpu information
    gpu_info = get_gpu_info(device_index)
    if not gpu_info:
        # if failed, shutdown nvml and exit
        pynvml.nvmlShutdown()
        return None
    
    # extract the gpu handle for queries
    handle = gpu_info['handle']
    # take a single measurement of memory usage
    mem_usage = get_gpu_memory_usage(handle)
    # take a single measurement of gpu utilization
    gpu_util = get_gpu_utilization(handle)
    
    # clean up: shutdown nvml connection
    pynvml.nvmlShutdown()
    
    # return all measurements as a dictionary with timestamp
    return {
        "gpu_name": gpu_info['name'],
        "memory": mem_usage,
        "gpu_utilization": gpu_util,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # this block runs when you execute: python gpu_monitor.py
    
    # create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = f"logs/gpu_monitor_{timestamp}.log"
    
    print("GPU Memory Monitor")
    print(f"Logging to: {log_file_path}")
    print("Press Ctrl+C to stop monitoring\n")
    
    # start continuous monitoring:
    # - device_index=0: monitor the first gpu (gpu 0)
    # - interval=0.25: take measurements every 0.25 seconds
    # - log_file: write output to timestamped log file in logs/ directory
    # run this in a separate terminal while vllm inference is running
    monitor_gpu_continuous(device_index=0, interval=0.25, log_file=log_file_path)
