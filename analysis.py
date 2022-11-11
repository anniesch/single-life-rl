import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import pickle
import os
import sys
import glob
import re
from collections import defaultdict

def collect_results(args):
    regex = args.save_dir
    filenames = glob.glob(regex)
    base = args.save_dir[:-2]
    filenames = [filename.split('/')[-1] for filename in filenames]
    exp_settings = sorted(filenames, key=lambda x: x[1])

    total_steps = defaultdict(list)
    for exp_name in exp_settings:
        try:
            with open(f"{base}/{exp_name}/total_steps.txt", 'r') as f:
                steps = f.read()
            setting_name = exp_name[:-1]
            total_steps[setting_name].append(int(steps))
        except:
            print(f"Can't read {exp_name}!")
                                
    for key in sorted(total_steps.keys()):
        vals = total_steps[key]
        N = len(vals)
        avg_string = f"{np.mean(vals):.3f} +- {np.std(vals):.3f} w/ stderr: {(np.std(vals))/np.sqrt(len(vals)):.3f}, median: {np.median(vals):.3f}"
        print(
            f"{os.path.basename(key):<45}\t{N}  {vals}"
        )
        print(
            f"Average:  {avg_string}"
        )
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="2022.11.05")
    parser.add_argument("--env_name", type=str, default="pointmass")
    parser.add_argument("--y_lim", type=int, default=200000)
    
    args = parser.parse_args()
    
    args.save_dir = f'exp_local/{args.env_name}/{args.date}/*'
    collect_results(args)