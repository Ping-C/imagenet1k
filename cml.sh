#!/bin/bash

# CML SETUP


# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=interactive                              # sets the job name if not set from environment
#SBATCH --array=0                    # Submit 8 array jobs, throttling to 4 at a time
#SBATCH --output inter_logs/session_%A_%a.log                            # indicates a file to redirect STDOUT to; %j is the jobid, _%A_%a is array task id
#SBATCH --error inter_logs/session_%A_%a.log                             # indicates a file to redirect STDERR to; %j is the jobid,_%A_%a is array task id
#SBATCH --account=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --partition=hipri
#SBATCH --nice=0                                              #positive means lower priority

sleep 6000000000


