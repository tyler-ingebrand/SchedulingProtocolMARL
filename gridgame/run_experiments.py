import os
from multiprocessing import Pool, freeze_support
import time
from test import run_experiment

if __name__ == "__main__":
    freeze_support()
    protocols = ['RR'] # ['CSMA_CD', 'STR', 'RR', 'NoneBlind', 'NoneSeeing']
    seeds = 5
    number_steps= 5000_000
    multi_process = True
    max_processes_at_once = 25
    os.makedirs("results", exist_ok=True)
    number_GPUs = 1
    currentGPU = 0

    if not multi_process:
        for p in protocols:
             for s in range(seeds):
                 run_experiment(p, s, number_steps=number_steps,)
    else:
        args = []
        for p in protocols:
            for s in range(seeds):
                this_arg = (p, s, number_steps, "cpu")
                args.append(this_arg)
                currentGPU += 1
                if currentGPU == number_GPUs:
                    currentGPU = 0

        while len(args) > 0:
            sub_args = [args.pop() for i in range(min(max_processes_at_once, len(args)))]
            with Pool(max_processes_at_once) as p:
                p.starmap(run_experiment, sub_args)

    print("\n\n\n\n\n\n\nAll done")