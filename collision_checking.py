import argparse as ap
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import csv
from utils import *
import json


parser = ap.ArgumentParser()
parser.add_argument("--robot")
parser.add_argument("--map")
args = parser.parse_args()
print(args)

def visualize_scene_collision(env, collision):
    plt.plot()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axis("equal")
    ax = plt.gca()
    for i, r in enumerate(env.obstacles):
        rv = rect_to_vertices(r)
        color = None
        if collision[i]:
            color = "red"
            print(f"Collision!: {rv}")
        else:
            color = "green"
        ax.add_patch(matplotlib.patches.Rectangle((rv[3]), r[3], r[4], angle=r[2]*180/np.pi, facecolor=color, edgecolor=("white", 0.0)))


def arm_collision2(q, obstacle):
    # Assume length of arm is 1, 
    r_actor = (np.cos(q[0])/2, np.sin(q[0])/2, q[0], 1, 0.1)
    r_actor2 = (np.cos(q[0]) + np.cos(q[0] + q[1])/2,
                np.sin(q[0]) + np.sin(q[0] + q[1])/2,
                q[0] + q[1], 1, 0.1)
    cond = collision_check(r_actor, [obstacle]) or collision_check(r_actor2, [obstacle])
    return cond

def freebody2D_collision2(q, obstacle):
    r_actor = (q[0], q[1], q[2], 0.5, 0.3)
    return collision_check(r_actor, [obstacle])

env = scene_from_file(args.map)

sample_fn = None
goal = None
collision_fn = None
vis_fn = None

if args.robot == "arm":
    sample_fn = arm_sample_conf
    goal = np.random.rand(2) * 2*np.pi 
    collision_fn = arm_collision2
    vis_fn = visualize_arm_conf
else:
    sample_fn = freebody2D_sample_conf
    goal = np.random.rand(3) * 2*np.pi # here just to make the function work, ignore
    collision_fn = freebody2D_collision2
    vis_fn = visualize_freebody_conf

def visualize_collider(r):
    rv = rect_to_vertices(r)
    x = [v[0] for v in rv[1:]]
    y=  [v[1] for v in rv[1:]]
    plt.fill(x, y, fill=False, edgecolor="red")

for i in range(10):
    conf = sample_fn(goal)
    r = [conf[0], conf[1], conf[2], 0.5, 0.3]
    rv = rect_to_vertices(r)
    while np.any(np.abs(rv) > 10):
        conf = sample_fn(goal)
        r = [conf[0], conf[1], conf[2], 0.5, 0.3]
        rv = rect_to_vertices(r)
    collisions = []
    for obstacle in env.obstacles:
        collisions.append(collision_fn(conf, obstacle))
    print(collisions)
    if np.any(collisions):
        print(f"Rv: {rv}")
    visualize_scene_collision(env, collisions)
    visualize_collider(r)
    vis_fn(conf)
    plt.pause(1.0)
    plt.savefig(f"{args.robot}_{args.map}_collision{i:03}.png")
    plt.clf()
