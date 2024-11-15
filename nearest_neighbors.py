import argparse as ap
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import csv
from utils import *


parser = ap.ArgumentParser()
parser.add_argument("--robot")
parser.add_argument("--target", nargs='+')
parser.add_argument("-k")
parser.add_argument("--configs")
parser.add_argument("--save", action="store_true")
parser.add_argument("--prefix", required=False, default=f"knn_default{np.random.randint(1000)}")
args = parser.parse_args()
print(args)

configurations = []
target_conf = np.array([float(x) for x in args.target])
with open(args.configs, "r") as f:
    reader = csv.reader(f, delimiter=" ") # Assuming config.txt is a space separated list
    for row in reader:
        print(row)
        configurations.append(np.array([float(q) for q in row]))

def Knn(target, confs, metric, k):
    sorted_confs = sorted(confs, key=lambda x: metric(target, x))
    return sorted_confs[:k]

metric_fn = None
vis_fn = None
if args.robot == "arm":
    metric_fn = arm_metric
    vis_fn = visualize_arm_conf
    if len(args.target) != 2:
        print("Mismatched target length!")
else:
    metric_fn = freebody2D_metric
    vis_fn = visualize_freebody_conf
    if len(args.target) != 3:
        print("Mismatched target length!")


nearest = Knn(target_conf, configurations, metric_fn, int(args.k))
print(nearest)
vis_fn(target_conf, color="orange")
for conf in nearest:
    vis_fn(conf)
if args.save:
    plt.savefig(f"{args.prefix}_knn.png")
plt.show()
