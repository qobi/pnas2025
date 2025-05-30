from multiprocessing import Process, Queue, Pool, Pipe, cpu_count
import numpy as np
import torch
from tqdm import tqdm

def collect_jobs(job_queue, out, sender):
    n_jobs = len(out)
    completed = 0
    with tqdm(total=n_jobs, desc="Processing trials", unit="trials", leave=False) as pbar:
        while completed < n_jobs:
            job_idx, job = job_queue.get()
            out[job_idx] = job
            completed += 1
            pbar.update(1)
        pbar.close()

    out = np.array(out)
    sender.send(out)
    sender.close()

def run_jobs(func, data, args = (), max_jobs = 1):   
    data = data.numpy()
    out = [None for _ in range(len(data))]


    job_queue = Queue()
    sender, receiver = Pipe()
    job_collector = Process(target=collect_jobs, args=(job_queue, out, sender, ))
    job_collector.start() 
    
    if max_jobs == -1:
        max_jobs = cpu_count()

    max_jobs -= 1
    max_jobs = max(max_jobs, 1)
    with Pool(max_jobs) as p:   
        for trial_idx, trial in enumerate(data):
            p.apply_async(func, args = (trial, *args), 
                                callback = lambda x, trial_idx=trial_idx: job_queue.put((trial_idx, x)))
        p.close()
        p.join()

    out = receiver.recv()
    receiver.close()

    job_collector.terminate()

    return torch.tensor(out, dtype=torch.float32)