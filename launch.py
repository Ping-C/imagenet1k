import submitit
import torch
import random
import subprocess
from functools import partial
import time
import argparse
import os
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--command', default="python train_imagenet.py --config-file configs/vit_100cls.yaml --logging.folder outputs/test_submitit --logging.project_name test --dist.world_size=4 --data.num_workers=4", help='command to run')
parser.add_argument('--log_folder', default='slurm_log', help='folder to store outputs')
parser.add_argument('--time_limit_minutes', type=int, default=60*24, help='timelimit in minutes')
parser.add_argument('--test_node_failure', action='store_true', help='timelimit in minutes')
parser.add_argument('--test_nccl_failure', action='store_true', help='timelimit in minutes')
parser.add_argument('--partition', default='lowpri', choices=[ 'hipri', 'lowpri' ], help='partition to use' )
args = parser.parse_args()


def wrapper_function(command, world_size, port, test_node_failure):
    job_env = submitit.JobEnvironment()
    master_node = job_env.hostnames[0]
    print(f"""=============================
    There are {job_env.num_tasks} tasks in this job
    I'm the task #{job_env.local_rank} on the node {job_env.node}
    I'm the task #{job_env.global_rank} in the job
    I'm on node #{job_env.hostname}
    All the associated nodes are #{job_env.hostnames}
    Number of gpus available on the node is {torch.cuda.device_count()}
    MASTERNODE is {master_node}
    ============================""")
    if job_env.local_rank+test_node_failure >= torch.cuda.device_count() :
        print(f"Error in nodes due to unavailable gpus in node {job_env.hostname}")
        return ("NODE_ERROR", job_env.hostname)
    else:
        try:
            command += f" --dist.address={master_node} --dist.port={port} --dist.multinode=1"
            submitit.helpers.CommandFunction(command.split())()
            return ("SUCCESS", None)
        except Exception as e:
            if "a hack to terminate the process" in str(e):
                return ("SUCCESS", None)
            print(f"Encountering unexpected failure: {e.__class__}{e}")
            return ("UNEXPECTED_SCRIPT_FAILURE", e)

def readlogfiles(job):
    # read log files if they exists
    # loop through all log files, and make sure that the error logs
    NCCL_ERR_MSG = "ncclSystemError: System call (socket, malloc, munmap, etc) failed."
    UNEXPECTED_ERR_MSG = "Encountering unexpected failure:"
    UNEXPECTED_ERR_MSG2 = "free(): double free detected in"
    statuses = [None]*len(job._tasks)
    for i, task_num in enumerate(job._tasks):
        # check whether the error log exists
        # if exists try to read it
        error_log_file = os.path.join(job._paths._folder, f"{job._job_id}_{task_num}_log.err")
        if os.path.exists(error_log_file):
            f = open(error_log_file, "r")
            lines = f.readlines()
            f.close()
            
            for line in lines:
                if NCCL_ERR_MSG in line or UNEXPECTED_ERR_MSG in line or UNEXPECTED_ERR_MSG2 in line:
                    statuses[i] = "NCCL_ERR"
                    break
                if "iter=1" in line:
                    statuses[i] = "TRAINING"
                    break
    return statuses
            
            # open the error log
def getnodefromtask(task):
    try:
        lines = task.stdout().split("\n")
        for line in lines:
            if "I'm on node #" in line:
                nodename = line.replace("    I'm on node #", "")
                return nodename
    except Exception as e:
        pass
    return ""
class SlurmExecutorMod(submitit.SlurmExecutor):
    def _num_tasks(self) -> int:
        additional_para = self.parameters.get("additional_parameters", None)
        if 'ntasks' in additional_para:
            return additional_para['ntasks']
        else:
            nodes: int = self.parameters.get("nodes", 1)
            tasks_per_node: int = max(1, self.parameters.get("ntasks_per_node", 1))
            return nodes * tasks_per_node
    
executor = SlurmExecutorMod(folder=args.log_folder)

world_size = int([val for val in args.command.split() if '--dist.world_size=' in val][0].split('=')[1])
max_minutes = args.time_limit_minutes # TODO: needs to change this to 60*24

port = random.randint(10000,30000)
function = partial(wrapper_function, args.command, world_size, port, args.test_node_failure)
excluded_nodes = []
requeue = True
requeue_known_count = 0
requeue_unknown_count = 0
requeue_limit = 3
while requeue and requeue_unknown_count < requeue_limit:
    print(f"""
    ***********************
    start launching jobs on the {requeue_known_count + requeue_unknown_count} try (known failure: {requeue_known_count}, unknown failure: {requeue_unknown_count})...
    running command: 
    {args.command}
    
    world_size: {world_size}
    excluded nodes: {excluded_nodes}
    ***********************""")
    
    
    
    
    executor.update_parameters(job_name='img1k', time=max_minutes, partition=args.partition, cpus_per_task=8, additional_parameters={'ntasks': world_size, 'gpus_per_task': 1, 'account': 'all'}, exclude=",".join(excluded_nodes), nodes='1-8')
    job = executor.submit(function)  # just a list of jobs
    # there are different failure cases that we need to catch and deal with
    # monitor the jobs, until all tasks are at least launched, and maybe monitor the progress?
    job
    tasks = [job.task(i) for i in range(world_size)]
    i = 0
    while True:
        time.sleep(10)
        # monitor the jobs by reading the logs. If nccl exception has occured in the particular 
        # check whether the file contain this error ("ncclSystemError: System call (socket, malloc, munmap, etc) failed.")
        if i % 12 == 0:
            statuses_by_log = readlogfiles(job)
            print(f"STATUSES BY LOG:{statuses_by_log}")
        
        # if "NCCL_ERR" in statuses_by_log:
        if "NCCL_ERR" in statuses_by_log or ( "TRAINING" in statuses_by_log and args.test_nccl_failure):
            print("encountering nccl_err, cancel job and restart")
            job.cancel()
            # ensure that the state has changed
            while "CANCELLED" not in job.state and "COMPLETED" not in job.state:
                print("trying to cancel job", job.state)
                time.sleep(10)
            
            break
            # cancel jobs
            
        print(f"job {job.job_id} states: ", [task.state for task in tasks])
        if all([task.state == 'COMPLETED' or task.state == 'FAILED' or task.state == 'NODE_FAIL' or task.state == 'TIMEOUT' for task in tasks]):
            break
            # start analyzing the results only if all of the tasks have either failed, completed or timeout
        if any(["CANCELLED" in task.state for task in tasks]):
            print("slurm job has been cancelled")
            exit()
        i += 1


    job_statuses = []
    job_exceptions = []
    for task_i, task in enumerate(tasks):
        if task.state == "FAILED":
            if "timed-out" in str(task.exception()):
                job_statuses.append("TIMEOUT")
                job_exceptions.append(None)
            else:
                job_statuses.append("UNKNOWN_SLURM_FAILURE")
                job_exceptions.append(task.exception())
        elif task.state == "TIMEOUT":
            job_statuses.append("TIMEOUT")
            job_exceptions.append(None)
        elif task.state == "COMPLETED":
            task_status, task_exception = task.result()
            job_statuses.append(task_status) # this is either hostname, SUCCESS, UNEXPECTED_SCRIPT_FAILURE
            job_exceptions.append(task_exception)
        elif "CANCELLED" in task.state:
            if statuses_by_log[task_i] == "NCCL_ERR" or args.test_nccl_failure:
                job_statuses.append("NODE_ERROR")
                job_exceptions.append(getnodefromtask(task))
            else:
                job_statuses.append("CANCELLED")
                job_exceptions.append(getnodefromtask(task))
        elif "NODE_FAIL" == task.state:
            job_statuses.append("NODE_ERROR")
            job_exceptions.append(getnodefromtask(task))
        else:
            raise ValueError(f"Unknown state {task.state}")
    print(f"job statuses: {job_statuses}")
    for i, job_exception in enumerate(job_exceptions):
        if job_exception is not None:
            print(f"Task {i} has the following exception", "="*20)
            print(job_exception)
    # if jobs failures are either timeout or unknown slurm failure, then requeue jobs
    
    if all([status == "SUCCESS"  for status in job_statuses]):
        requeue = False
    elif any([status == "TIMEOUT" for status in job_statuses]):
        requeue = True
        requeue_known_count += 1
    elif any([status in ("NODE_ERROR", "NODE_FAILED", "NODE_FAIL")  for status in job_statuses]):
        # if faulty nodes are returned
        requeue = True
        requeue_known_count += 1
        failed_nodes = [ exception for status, exception in zip(job_statuses, job_exceptions) if status in ("NODE_ERROR", "NODE_FAILED", "NODE_FAILED") ]
        excluded_nodes += list(set(failed_nodes))
    else:
        # try requeuing no more than 3 times, if the error is unclear
        requeue = True
        requeue_unknown_count += 1



    # 1. if we encounter bad nodes, then we should restart the job (This exception should be dealt with inside of the function)
    # 2. if the job timed out, then we should restart the job (This exception should be delth with using the status of the job)
    # 3. if the job finished successfully, then we should not relaunch the job (This is not an exception, and should be good)
    # 4. if the job has other error (This should be catched inside of the function itself)
    
    
    
    