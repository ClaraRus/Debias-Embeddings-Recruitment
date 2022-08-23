import os
import json
from argparse import ArgumentParser
from pipeline import Pipeline


parser = ArgumentParser()
parser.add_argument("--path_configs",
                    help="Path to folder containing the configs files", )
parser.add_argument("--experiments", help="No of experiments to run")
parser.add_argument("--runs", help="No of runs per experiments")

args = parser.parse_args()

path_configs = args.path_configs #'configs/'
experiments = int(args.experiments) #2
runs = int(args.runs) #5

configs = []
for experiment in range(1,experiments+1):
    path_experiment = os.path.join(path_configs, 'EXPERIMENT_' + str(experiment))
    configs_experiments = os.listdir(path_experiment)
    
    configs_experiments = [os.path.join(path_experiment, c) for c in configs_experiments]
    configs.extend(configs_experiments)
    

for config_file in configs:
    if '.json' in config_file:
        print(config_file)
        for run in range(0, runs):
            pipeline = Pipeline(config_file, run)
            pipeline.start(verbose=True)