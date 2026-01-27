# NVIDIA Nsight Systems GPU Profiling Demo

## Quick Start

1. Create and activate virtual environment: `python3 -m venv venv && source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Profile with Nsight Systems: `nsys profile -o torch_profile python torch_demo.py`
4. View results: `nsys-ui torch_profile.nsys-rep` or open the `.nsys-rep` file in Nsight Systems GUI
5. Look for NVTX markers ("Data Initialization", "Matrix Multiplication Loop") in the timeline to track GPU performance
