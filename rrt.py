import argparse as ap
from utils import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import json
from math import *

class TreeNodeKD:
    def __init__(self, location):
        self.location = location
        self.children = []
        self.velocity = np.zeros(2)
        self.parent = None
        self.control = None

class RRT:
    def __init__(self, start, goal_region, env, sample_conf, metric, expand, collision, collision_path, diff_func):
        self.start = start
        self.goal = np.array(goal_region) # (q_goal, radius)
        self.tree = TreeNodeKD(self.start)
        self.nodes = [self.tree]
        self.sample_conf = sample_conf
        self.metric = metric
        self.expand = expand
        self.collision = collision
        self.collision_path = collision_path
        self.diff_func = diff_func
        self.env = env

    def step(self, dt=0.1, visualize=False):
        x_rand = self.sample_conf(self.goal)
        x_near_node = self.nearest(x_rand)
        x_near = x_near_node.location # Extract location from tree
        # u = self.sample_ctrl(x_near, x_rand, x_near_node.velocity, dt) # Since we have a very simple 2D robot, we just use the displacement vector to create our accel vector
        # x_new, v_new = self.simulate(x_near, x_near_node.velocity, u, dt)
        x_new = self.expand(x_near, x_rand, alpha=0.2)
        if not self.collision_path(x_near, x_new, self.env):
            x_new_node = TreeNodeKD(x_new)
            x_new_node.parent = x_near_node
            x_near_node.children.append(x_new_node)
            self.nodes.append(x_new_node)
            if self.in_goal(x_new):
                print("Goal reached!")
                return True, x_new_node
            if visualize:
                plt.pause(0.1)
                plt.plot([x_near[0], x_new[0]], [x_near[1], x_new[1]], "black", marker=".", linestyle="--",)
        return False, None
  
    def in_goal(self, loc):
        radius = self.goal[-1]
        if np.abs(self.metric(loc, self.goal[:-1])) <= radius:
            return True
        return False

    def nearest(self, x_rand):
        nearest_node = self.tree
        nearest_dist = self.metric(nearest_node.location, x_rand)
        for n in self.nodes:
            if self.metric(x_rand, n.location) < nearest_dist:
                nearest_dist = self.metric(x_rand, n.location)
                nearest_node = n
        return nearest_node
    
    def visualize(self, vtype="freebody"):
        if vtype == "freebody":
            visualize_scene(self.env)
            ax = plt.gca()
            fringe = [self.tree]
            visited = set()
            while len(fringe) != 0:
                node = fringe[0]
                fringe = fringe[1:]
                if node in visited:
                    continue
                visited.add(node)
                fringe.extend(node.children)
                ax.add_patch(matplotlib.patches.Circle(node.location[:2], radius=0.01))
                if node.parent is not None:
                    p = node.parent
                    diff = node.location - p.location
                    ax.add_patch(matplotlib.patches.Arrow(p.location[0], p.location[1], diff[0], diff[1], width=0.02, linestyle="--", linewidth=0.02))
        elif vtype == "arm":
            visualize_arm_cspace(self.env)
            x = np.linspace(0, 2*np.pi, 100)
            y = np.linspace(0, 2*np.pi, 100)
            plt.plot()
            plt.xlim(0, 2*np.pi)
            plt.ylim(0, 2*np.pi)
            plt.axis("equal")
            ax = plt.gca()
            for j in y:
                for i in x:
                    if self.collision(np.array([i, j]), self.env):
                        ax.add_patch(matplotlib.patches.Circle((i, j), radius=0.02, color="red"))
            
            fringe = [self.tree]
            visited = set()
            while len(fringe) != 0:
                node = fringe[0]
                fringe = fringe[1:]
                if node in visited:
                    continue
                visited.add(node)
                fringe.extend(node.children)
                ax.add_patch(matplotlib.patches.Circle([constrain_angle(i) for i in node.location], radius=0.01))
                if node.parent is not None:
                    p = node.parent.location
                    p = [constrain_angle(i) for i in p]
                    diff = [constrain_angle(node.location[i]) - p[i] for i in range(2)]
                    ax.add_patch(matplotlib.patches.Arrow(p[0], p[1], diff[0], diff[1], width=0.02, linestyle="--", linewidth=0.2))
            
if __name__ == "__main__":            
    parser = ap.ArgumentParser()
    parser.add_argument("--robot")
    parser.add_argument("--start", nargs="+")
    parser.add_argument("--goal", nargs="+")
    parser.add_argument("--goal_rad")
    parser.add_argument("--map")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--prefix", required=False, default=f"rrt_default{np.random.randint(1000)}")
    args = parser.parse_args()
    
    start = np.array([float(qi) for qi in args.start])
    goal = np.array([float(qi) for qi in args.goal] + [float(args.goal_rad)])
    env = scene_from_file(args.map)
    
    rrt = None
    vis_type = None
    
    if args.robot == "arm":
        rrt = RRT(start, goal, env, arm_sample_conf, arm_metric,
                  arm_expand, arm_collision, arm_collision_path, arm_difference)
        vis_type = "arm"
    else:
        rrt = RRT(start, goal, env, freebody2D_sample_conf, freebody2D_metric, freebody2D_expand,
              freebody2D_collision, freebody2D_collision_path, freebody2D_difference)
        vis_type = "freebody"
    
    if args.robot == "arm":
        visualize_arm_cspace(env)
    else:
        visualize_scene(env)
    frno = 0
    while len(rrt.nodes) < 1000:
        c, r = rrt.step(visualize=True)
        if args.save:
            plt.savefig(f"{args.prefix}_treegrowth_fr{frno:04}.png")
        else:
            plt.pause(0.03)
        if c:
            print("Goal Reached!")
            break
        print(f"Node count: {len(rrt.nodes)}")
        frno += 1
    
    if not c:
        print("No viable path found!")
        exit(-1)
    plt.show()
    
    path = []
    while r is not None:
        path.append(r.location)
        r = r.parent
    path = path[::-1]
    print(path)
    visualize_path(rrt, vis_type, path)
    if args.save:
        plt.savefig(f"{args.prefix}_path.png")
    plt.show()
    if args.robot == "arm":
        animate_arm(env, path, interp=True, savefig=args.save, prefix=f"{args.prefix}_arm_anim")
    else:
        animate_freebody(env, path, interp=False, savefig=args.save, prefix=f"{args.prefix}_freebody_anim")
