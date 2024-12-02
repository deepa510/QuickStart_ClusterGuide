# QuickStart Guide for Connecting to SLURM Cluster

**Objective**: This document provides a stepwise guide for connecting to SLURM.

**Note**: In order to connect, please make sure to connect to **USF VPN**.

---
## Accounts
-> Please note: CIRCE is for research-related use only.
To request an account on CIRCE, please send an email (from your official USF email address) to rc-help@usf.edu with the subject "CIRCE Account Request". In this email, please also provide the following info:

- **Faculty Sponsor Full Name:**
- **Faculty Sponsor Department:**
- **Faculty Sponsor USF Email Address:**

## Connecting & Accessing
- **To connect to CIRCE, you will need to use an SSH client or utilize the CIRCE Desktop Environment.**

## Connecting via SSH

The following information will be needed to connect via SSH:

- **Your USF NetID and Password**
- **Hostname**: `circe.rc.usf.edu`
- **SSH Port**: 22 (default)

### SSH Clients for Windows

- **PuTTY**: [PuTTY Download Link](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)

### SSH Clients for Mac OSX

- **OSX SSH Tutorial**: ([https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html](https://osxdaily.com/2017/04/28/howto-ssh-client-mac/))

## Check Available Partitions:
**sinfo** -This command will display information about partitions, nodes, and their status.

![image](https://github.com/user-attachments/assets/655f88c2-bda3-441b-953f-2b9e4671057c)


## View Partition Details:
**sinfo -p muma_2021** -The -p option specifies the partition for which you want information, in this case, the muma_2021 partition.

![image](https://github.com/user-attachments/assets/c85d37fe-2083-400b-8b49-5324dc726c9e)

## Getting to know TERMS :

1) **PARTITION**  
   This column shows the name of the partition (logical grouping of nodes) in the Slurm cluster.

2) **AVAIL**  
   This column indicates whether the partition is available for use.  
   - The value `up` means the partition is online and available to accept jobs.  
   - If it were `down`, that would indicate that the partition is unavailable.

3) **TIMELIMIT**  
   This column shows the maximum time limit that jobs are allowed to run on this partition.  
   - The value `infinite` means there is no time limit for jobs on this partition.  
   - Jobs can run indefinitely unless explicitly restricted in the job script.

4) **NODES**  
   This column shows the number of nodes (physical machines or servers) in the partition that are in the specified state.  
   - In the first row, 3 nodes are in the `drng` (draining) state.  
   - In the second row, 7 nodes are in the `idle` state.

5) **STATE**  
   This column describes the current state of the nodes in the partition:  
   - `drng` (draining): The 3 nodes listed (mdc-1057-18-[1-2,4]) are in the process of being taken offline. These nodes are finishing the jobs they are currently running but will not accept new jobs.  
   - `idle`: The 7 nodes listed (mdc-1057-13-[8-13], mdc-1057-18-3) are currently idle and available to run new jobs. No jobs are currently running on these nodes.

6) **NODELIST**  
   This column lists the names or ranges of the nodes in the partition, which are in the given state.

## Digging INTO PARTITION
**scontrol show partition muma_2021**

![image](https://github.com/user-attachments/assets/ca38c3e1-3a4a-4c1e-9f22-afd9339d4427)

## 1. **Partition Name:**
PartitionName=muma_2021
- The partition is named **`muma_2021`**, which is a specific grouping of nodes designated for job submissions.

## 2. **Access Control:**
AllowGroups=ALL AllowAccounts=ALL AllowQos=trouble-shooting,muma21,preempt_short
- **AllowGroups=ALL**: All user groups are allowed to submit jobs to this partition.
- **AllowAccounts=ALL**: All user accounts are allowed to submit jobs to this partition.
- **AllowQos=trouble-shooting,muma21,preempt_short**: The partition supports multiple **Quality of Service (QoS)** levels, which influence job priority and behavior. These QoS levels are:
  - **trouble-shooting**: Likely used for debugging or fixing issues.
  - **muma21**: A custom QoS likely set for a specific project or user group.
  - **preempt_short**: Used for short jobs that can be preempted by higher priority jobs.

## 3. **Resource Allocation:**
AllocNodes=ALL Default=NO QoS=N/A
- **AllocNodes=ALL**: All nodes in the partition are available for job allocation.
- **Default=NO**: This partition is not the default partition for job submissions.
- **QoS=N/A**: There is no default QoS for this partition; users must explicitly specify QoS if required.

## 4. **Time Limits and Root Jobs:**
DefaultTime=01:00:00 DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO
- **DefaultTime=01:00:00**: If no time limit is specified in the job submission, the default time limit is set to **1 hour**.
- **DisableRootJobs=NO**: Root users are allowed to submit jobs.
- **ExclusiveUser=NO**: The partition is not restricted to specific users; any authorized user can submit jobs.
- **GraceTime=0**: No grace period is allowed for jobs in this partition.
- **Hidden=NO**: The partition is not hidden and is visible to all users.

## 5. **Node Configuration:**
MaxNodes=UNLIMITED MaxTime=UNLIMITED MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED
- **MaxNodes=UNLIMITED**: There is no limit on the number of nodes that can be used for a job.
- **MaxTime=UNLIMITED**: There is no maximum runtime limit for jobs submitted to this partition.
- **MinNodes=0**: Jobs can run on as few as 0 nodes, which is usually relevant for jobs that are pending.
- **LLN=NO**: **LLN** stands for "Lowest Level Nodes." In this case, it's set to **NO**, meaning there are no specific node constraints.
- **MaxCPUsPerNode=UNLIMITED**: There is no upper limit to the number of CPUs that can be allocated per node.

## 6. **Node List:**
Nodes=mdc-1057-13-[9-13],mdc-1057-18-[1-4],mdc-1057-13-8
- The nodes in this partition include:
  - **mdc-1057-13-[9-13]**: Nodes numbered 9 to 13 on this system.
  - **mdc-1057-18-[1-4]**: Nodes numbered 1 to 4 on this system.
  - **mdc-1057-13-8**: Node 8.

## Types of Nodes in Computing Clusters

#### 1. **Login Nodes**:
   - **Purpose**: These nodes are used for users to log into the system, compile programs, submit jobs, and perform light preparatory tasks. Users don't run intensive computations on these nodes.
   - **Example**: Users connect to the login node to interact with the cluster, but computation is submitted to compute nodes.

#### 2. **Compute Nodes**:
   - **Purpose**: These are the nodes where the actual computations or job processing happens. When a job is submitted, it runs on compute nodes.
   - **Characteristics**: 
     - Typically have a large number of CPUs and GPUs.
     - Optimized for high-performance tasks.
   - **Example**: A job requesting several cores or GPUs will run on these nodes.

#### 3. **Master Nodes (Head Nodes)**:
   - **Purpose**: The central node that manages the cluster and handles scheduling, job management, and monitoring. It distributes tasks to compute nodes.
   - **Characteristics**: 
     - High memory for job scheduling and cluster management.
     - Doesn’t usually perform heavy computation itself.
   - **Example**: SLURM or other scheduling software often runs on the master node.

#### 4. **GPU Nodes**:
   - **Purpose**: These nodes are equipped with one or more Graphics Processing Units (GPUs) for tasks requiring high-performance parallel processing.
   - **Characteristics**: 
     - Commonly used for AI, machine learning, or high-end graphical computations.
     - GPUs allow faster processing for parallel workloads.
   - **Example**: TensorFlow or PyTorch jobs may use GPU nodes for deep learning.

#### 5. **Data Nodes**:
   - **Purpose**: Nodes designed specifically for handling large volumes of data, such as reading from or writing to storage devices.
   - **Characteristics**: 
     - Optimized for I/O-heavy operations.
     - Often connected to storage arrays or large disk systems.
   - **Example**: Nodes used for data preprocessing or storing large datasets in distributed computing environments.

#### 6. **Storage Nodes**:
   - **Purpose**: Nodes dedicated to handling and managing storage. These nodes manage access to large volumes of data stored in distributed file systems like Lustre or HDFS.
   - **Characteristics**: 
     - Large storage capacity and fast I/O operations.
   - **Example**: Handling distributed storage systems like HDFS in big data processing.

## 7. **Job Scheduling and Priority:**
PriorityJobFactor=1 PriorityTier=1 RootOnly=NO ReqResv=NO OverSubscribe=NO
- **PriorityJobFactor=1**: Jobs submitted to this partition have a neutral priority factor (no special prioritization).
- **PriorityTier=1**: This partition is in the lowest priority tier (priority tiers define the order in which jobs are scheduled).
- **RootOnly=NO**: Non-root users are allowed to submit jobs.
- **ReqResv=NO**: Job reservations are not required to run in this partition.
- **OverSubscribe=NO**: Oversubscription is not allowed, meaning jobs cannot request more resources than are physically available on a node.

## 8. **Job Preemption and Limits:**
OverTimeLimit=NONE PreemptMode=REQUEUE
- **OverTimeLimit=NONE**: Jobs cannot exceed their allocated time limit.
- **PreemptMode=REQUEUE**: If a job is preempted, it will be requeued rather than canceled.

## 9. **Partition Status and Resources:**

State=UP TotalCPUs=928 TotalNodes=10 SelectTypeParameters=NONE
- **State=UP**: The partition is active and available for job submission.
- **TotalCPUs=928**: There are 928 CPUs available across all nodes in this partition.
- **TotalNodes=10**: There are 10 nodes available in the partition.
- **SelectTypeParameters=NONE**: There are no special selection parameters for choosing nodes.

## 10. **Job Defaults and Memory Settings:**

JobDefaults=(null)
DefMemPerCPU=512 MaxMemPerNode=UNLIMITED
- **JobDefaults=(null)**: No specific job defaults are configured for this partition.
- **DefMemPerCPU=512**: The default memory per CPU is **512 MB**.
- **MaxMemPerNode=UNLIMITED**: There is no upper memory limit per node; jobs can use as much memory as is physically available on the node.

## HOW to run a JOB in Cluster:

### Before running a job we need to create conda environment 
---
************************ Instructions on installing Miniconda3 *************************************

```bash
cd $HOME
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
when the installer asks:

Do you wish the installer to initialize Miniconda3 by running conda init?
please answer "yes"

log-out of CIRCE/RRA, then log back in
which conda pip
Make sure it returns with:

~/miniconda3/bin/conda
~/miniconda3/bin/pip

pip install --upgrade pip
conda update --all -y
conda install -y anaconda
Once Conda is installed, you can then create virtual environments for each package (or combination of packages) that you may need, including Tensorflow, Keras, OpenCV, gurobipy, numpy, etc.
```
Once your environment is ready we can proceed to begin activating and installing packages 
---
## 1. Connect to Cluster

![image](https://github.com/user-attachments/assets/b4e6d187-25a7-44ca-bac7-acabe7bfa690)

### 1.1 Request GPU Resource
```bash
srun -p muma_2021 -q muma21 --gres=gpu:1 --pty /bin/bash
```
![image](https://github.com/user-attachments/assets/e99fda59-aece-44e2-8f7e-500989924858)

### 1.2. Activate Environment
```bash
conda activate myenv
```
![image](https://github.com/user-attachments/assets/d18c8043-aff5-4b63-bec7-3db9d1d2de84)

### 1.3. Create Working Directory
```bash
mkdir pytorch_benchmark
cd pytorch_benchmark
```
![image](https://github.com/user-attachments/assets/c783b3e4-301c-4e67-afed-9f61548f7f3f)

## 2. Verify GPU Setup
### 2.1. Check GPU Status
```bash
nvidia-smi
```
![image](https://github.com/user-attachments/assets/268c1391-d2de-44e4-887d-b6055c37c08a)

Expected output should show:
- NVIDIA L40S GPU
- Available memory
- CUDA version

### 2.2. Install Required Package
```bash
pip install prettytable
```
![image](https://github.com/user-attachments/assets/3b5b1f91-4686-4bfb-ac46-838abef52810)


## 3. Create Benchmark Script

### 3.1. Create Python File
```bash
vi benchmark.py
```

### 3.2. Add Benchmark Code
This is a sample code that is a Python script benchmarks the performance of matrix multiplication and neural network forward passes on both CPU and GPU using PyTorch, comparing execution times and calculating speedups. The results are displayed in a formatted table for analysis.

```python
import torch
import torch.nn as nn
import time
import numpy as np
from prettytable import PrettyTable

class SimpleNN(nn.Module):
    def __init__(self, input_size=1000, hidden_size=100, output_size=10):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def benchmark_matrix_ops(size, device, num_iterations=10):
    # Matrix multiplication benchmark
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(5):
        _ = torch.mm(a, b)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(num_iterations):
        _ = torch.mm(a, b)
        if device == 'cuda':
            torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / num_iterations

def benchmark_neural_network(batch_size, input_size, device, num_iterations=10):
    # Neural network forward pass benchmark
    model = SimpleNN(input_size=input_size).to(device)
    data = torch.randn(batch_size, input_size, device=device)
    
    # Warmup
    for _ in range(5):
        _ = model(data)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(data)
        if device == 'cuda':
            torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / num_iterations

def run_benchmarks():
    # Initialize results table
    table = PrettyTable()
    table.field_names = ["Operation", "CPU Time (ms)", "GPU Time (ms)", "Speedup (x)"]
    
    # Matrix multiplication benchmarks with different sizes
    sizes = [1000, 2000, 4000]
    for size in sizes:
        print(f"\nRunning matrix multiplication benchmark with size {size}x{size}...")
        cpu_time = benchmark_matrix_ops(size, 'cpu') * 1000  # Convert to ms
        gpu_time = benchmark_matrix_ops(size, 'cuda') * 1000  # Convert to ms
        speedup = cpu_time / gpu_time
        table.add_row([f"Matrix Mult ({size}x{size})", 
                      f"{cpu_time:.2f}", 
                      f"{gpu_time:.2f}", 
                      f"{speedup:.2f}"])
    
    # Neural network benchmarks with different batch sizes
    input_size = 1000
    batch_sizes = [64, 128, 256]
    for batch_size in batch_sizes:
        print(f"\nRunning neural network benchmark with batch size {batch_size}...")
        cpu_time = benchmark_neural_network(batch_size, input_size, 'cpu') * 1000
        gpu_time = benchmark_neural_network(batch_size, input_size, 'cuda') * 1000
        speedup = cpu_time / gpu_time
        table.add_row([f"Neural Network (batch={batch_size})", 
                      f"{cpu_time:.2f}", 
                      f"{gpu_time:.2f}", 
                      f"{speedup:.2f}"])
    
    # Print summary
    print("\nBenchmark Results:")
    print(table)
    
    # Save results to file
    with open('benchmark_results.txt', 'w') as f:
        f.write("GPU vs CPU Benchmark Results\n")
        f.write("==========================\n")
        f.write("\nSystem Information:\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")
        f.write("\nDetailed Results:\n")
        f.write(str(table))

def main():
    print("Starting GPU vs CPU Performance Benchmark")
    print("========================================")
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        run_benchmarks()
    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
```

## 4. Run Benchmark

### 4.1. Execute Script
```bash
python benchmark.py
```

### 4.2. Expected Results
You should see output similar to:
```
Starting GPU vs CPU Performance Benchmark
========================================
PyTorch Version: 2.4.1+cu121
CUDA Available: True
GPU Device: NVIDIA L40S

Running matrix multiplication benchmark with size 1000x1000...
Running matrix multiplication benchmark with size 2000x2000...
Running matrix multiplication benchmark with size 4000x4000...
Running neural network benchmark with batch size 64...
Running neural network benchmark with batch size 128...
Running neural network benchmark with batch size 256...

Benchmark Results:
+----------------------------+---------------+---------------+-------------+
|         Operation          | CPU Time (ms) | GPU Time (ms) | Speedup (x) |
+----------------------------+---------------+---------------+-------------+
|  Matrix Mult (1000x1000)   |     12.75     |      0.09     |    139.53   |
|  Matrix Mult (2000x2000)   |     98.93     |      0.41     |    240.54   |
|  Matrix Mult (4000x4000)   |     825.83    |      2.82     |    292.49   |
| Neural Network (batch=64)  |      0.30     |      0.09     |     3.21    |
| Neural Network (batch=128) |      0.40     |      0.09     |     4.36    |
| Neural Network (batch=256) |      0.63     |      0.09     |     7.02    |
+----------------------------+---------------+---------------+-------------+
```

## 5. Key Points to Note

### 5.1. Matrix Multiplication Results
- Shows dramatic speedup with larger matrices
- 1000x1000: ~140x speedup
- 2000x2000: ~240x speedup
- 4000x4000: ~290x speedup

### 5.2. Neural Network Results
- Speedup increases with batch size
- Ranges from 3x to 7x speedup
- Shows GPU efficiency with parallel processing

### 5.3. Performance Factors
- Matrix size significantly impacts speedup
- Larger computations show greater GPU advantage
- Results saved in 'benchmark_results.txt' for reference

## 6. Cleanup

### 6.1. Exit Session
```bash
exit
```

### 6.2. For New Session
Repeat steps from Section 1 to start fresh   









