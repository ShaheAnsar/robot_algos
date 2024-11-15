from prm import PRM

import argparse as ap
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import csv
from utils import *
import json



#parser = ap.ArgumentParser()
#parser.add_argument("--robot")
#parser.add_argument("--start", nargs="+")
#parser.add_argument("--goal", nargs="+")
#parser.add_argument("--map")
#parser.add_argument("--goal_rad")
#args = parser.parse_args()

parser = ap.ArgumentParser()
parser.add_argument("--robot")
parser.add_argument("--start", nargs="+")
parser.add_argument("--goal", nargs="+")
parser.add_argument("--goal_rad")
parser.add_argument("--map")
parser.add_argument("--save", action="store_true")
parser.add_argument("--prefix", required=False, default=f"prmao_default{np.random.randint(1000)}")
args = parser.parse_args()
print(args)

start = np.array([float(qi) for qi in args.start])
goal = np.array([float(qi) for qi in args.goal] + [0.3])
env = scene_from_file(args.map)

prm = None
vis_type = None

if args.robot == "arm":
    prm = PRM(start, goal, env, arm_sample_conf, arm_metric,
              arm_expand, arm_collision, arm_collision_path, arm_difference, 2*np.pi/10, ao=True)
    vis_type = "arm"
else:
    prm = PRM(start, goal, env, freebody2D_sample_conf, freebody2D_metric, freebody2D_expand,
          freebody2D_collision, freebody2D_collision_path, freebody2D_difference, 1.0, ao=True)
    vis_type = "freebody"


while len(prm.nodes) < 5000:
    prm.step()
    print(f"Node count: {len(prm.nodes)}")

path = prm.a_star()
visualize_path(prm, vis_type, path)

if args.save:
    plt.savefig(f"{args.prefix}_{args.robot}_prm.png")
plt.show()
if args.robot == "arm":
    animate_arm(env, path, interp=True, savefig=args.save, prefix=f"{args.prefix}_arm_anim")
else:
    animate_freebody(env, path, interp=True, savefig=args.save, prefix=f"{args.prefix}_freebody_anim")
