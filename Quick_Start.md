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
-- **To connect to CIRCE, you will need to use an SSH client or utilize the CIRCE Desktop Environment.**

## Connecting via SSH

The following information will be needed to connect via SSH:

- **Your USF NetID and Password**
- **Hostname**: `circe.rc.usf.edu`
- **SSH Port**: 22 (default)

### SSH Clients for Windows

- **PuTTY**: [PuTTY Download Link](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)

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
> **Note**: Here we are taking an example to run a Data Analysis by loading spark.
## a. **Writing a JOB(.sh file):**
![image](https://github.com/user-attachments/assets/e98f6294-f3b7-4029-8053-0eb61ca313c9)
## b. **Submitting a JOB:**
- **sbatch job_script.sh**







