import submitit
import torch
import random
import subprocess
from functools import partial
import time
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--command', default="python train_imagenet.py --config-file configs/vit_100cls.yaml --logging.folder outputs/test_submitit --logging.project_name test --dist.world_size=4 --data.num_workers=4", help='command to run')
parser.add_argument('--log_folder', default='slurm_log', help='folder to store outputs')
parser.add_argument('--time_limit_minutes', type=int, default=60*24, help='timelimit in minutes')
parser.add_argument('--test_node_failure', action='store_true', help='timelimit in minutes')
parser.add_argument('--partition', default='lowpri', choices=[ 'hipri', 'lowpri' ], help='partition to use' )
args = parser.parse_args()


def wrapper_function(command, world_size, port, test_node_failure):
    job_env = submitit.JobEnvironment()
    
    print(f"There are {job_env.num_tasks} tasks in this job")
    print(f"I'm the task #{job_env.local_rank} on the node {job_env.node}")
    print(f"I'm the task #{job_env.global_rank} in the job")
    print(f"I'm on node #{job_env.hostname}")
    print(f"All the associated nodes are #{job_env.hostnames}")
    print(f"Number of gpus available on the node is {torch.cuda.device_count()}")
    master_node = job_env.hostnames[0]
    print(f"MASTERNODE is {master_node}")
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
    print(f"start launching jobs on the {requeue_known_count + requeue_unknown_count} try (known failure: {requeue_known_count}, unknown failure: {requeue_unknown_count})...")
    print(f"running command: {args.command}")
    print(f"world_size: {world_size}")
    print(f"excluded nodes: {excluded_nodes}")
    executor.update_parameters(job_name='img1k', time=max_minutes, partition=args.partition, cpus_per_task=8, additional_parameters={'ntasks': world_size, 'gpus_per_task': 1, 'account': 'all'}, exclude=",".join(excluded_nodes), nodes='1-8')
    job = executor.submit(function)  # just a list of jobs
    # there are different failure cases that we need to catch and deal with
    # monitor the jobs, until all tasks are at least launched, and maybe monitor the progress?
    tasks = [job.task(i) for i in range(world_size)]
    while True:
        time.sleep(10)
        print(f"job {job.job_id} states: ", [task.state for task in tasks])
        if all([task.state == 'COMPLETED' or task.state == 'FAILED' or task.state == 'TIMEOUT' for task in tasks]):
            break
            # start analyzing the results only if all of the tasks have either failed, completed or timeout
        if any(["CANCELLED" in task.state for task in tasks]):
            print("slurm job has been cancelled")
            exit()


    
    job_statuses = []
    job_exceptions = []
    for task in tasks:
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
        failed_nodes = [ exception for status, exception in zip(job_statuses, job_exceptions) if status in ("NODE_ERROR", "NODE_FAILED") ]
        excluded_nodes += failed_nodes
    else:
        # try requeuing no more than 3 times, if the error is unclear
        requeue = True
        requeue_unknown_count += 1



    # 1. if we encounter bad nodes, then we should restart the job (This exception should be dealt with inside of the function)
    # 2. if the job timed out, then we should restart the job (This exception should be delth with using the status of the job)
    # 3. if the job finished successfully, then we should not relaunch the job (This is not an exception, and should be good)
    # 4. if the job has other error (This should be catched inside of the function itself)
    
    
    
    