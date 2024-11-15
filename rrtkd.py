import argparse as ap
from utils import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import json
from math import *
from rrt import TreeNodeKD


def car_collision(q, env):
    r_actor = (q[0], q[1], q[2], 0.5, 0.8)
    obstacles = env.obstacles
    return collision_check(r_actor, obstacles)

def car_collision_path(q1, q2, env, n=10): # Use a simple raymarch to see if the path is viable
    qdiff = freebody2D_difference(q2, q1)
    if car_collision(q1, env) or car_collision(q2, env):
        return True # If either the start or the end points are in collision, then return true
    for i in range(1, n):
        qp = q1 + qdiff * i/n
        if car_collision(qp, env):
            return True
    return False # No collisions so far so return false

def rrt_get_controls(r):
    controls = []
    r2 = r
    while r2 is not None and r2.control is not None:
        print("works!")
        controls.append(r2.control)
        r2 = r2.parent
    controls = controls[::-1]
    return controls

def simulate(x_near, u, dt, simulate_factor=10):
    # Assume size of car is L=0.8,W=0.5
    x_new = np.array(x_near.tolist())
    steer_angle = np.arctan(0.5*tan(u[1]))
    for i in range(simulate_factor):
        x_new[0] += u[0]*cos(x_new[2] + steer_angle)*dt
        x_new[1] += u[0]*sin(x_new[2] + steer_angle)*dt
        x_new[2] += 2*u[0]/0.8 * sin(steer_angle) * dt
        x_new[2] = constrain_angle(x_new[2])
    return x_new


def simulate_analytical(x_near, u, dt, simulate_factor=10):
    # Assume size of car is L=0.8,W=0.5
    x_new = np.array(x_near.tolist())
    beta = np.arctan(0.5*tan(u[1]))
    v = u[0]
    omega = 2*v/0.8 * sin(beta)
    x_new[0] += v/omega*sin(x_new[2] + omega*dt*simulate_factor) - v/omega * sin(x_new[2])
    x_new[1] += -v/omega*cos(x_new[2] + omega*dt*simulate_factor) + v/omega * cos(x_new[2])
    x_new[2] += omega*dt*simulate_factor
    return x_new



def animate_car(env, start, controls, simulate_fn=simulate_analytical, dt=0.1, save=False,prefix="car_anim", simulate_factor=10):
    path = [start]
    x_curr = start
    for u in controls:
        for i in range(simulate_factor):
            x_curr = simulate_fn(x_curr, u, dt, simulate_factor=1) # Interpolation of sorts
            path.append(x_curr)
    for i, pose in enumerate(path):
        plt.clf()
        if car_collision(pose, env):
            print("oh no")
        visualize_scene(env)
        ax = plt.gca()
        rv = rect_to_vertices([pose[0], pose[1], pose[2], 0.8, 0.5])
        ax.add_patch(matplotlib.patches.Rectangle(rv[3], 0.8, 0.5, angle=pose[2] * 180/np.pi, color="gray"))
        if save:
            plt.savefig(f"{prefix}_fr{i:05}.png")
        plt.pause(0.1)


class KDRRT:
    def __init__(self, start, goal_region, env, sample_conf, metric, expand, collision, collision_path, diff_func, simulate, simulate_factor):
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
        self.simulate = simulate
        self.simulate_factor = simulate_factor # Chooses to simulate these many dts

    def step(self, dt=0.1, visualize=False):
        x_rand = self.sample_conf(self.goal)
        x_near_node = self.nearest(x_rand)
        x_near = x_near_node.location # Extract location from tree
        x_new, u = self.sample_ctrl(x_near, x_rand, dt)

        if not self.collision_path(x_near, x_new, env):
            x_new_node = TreeNodeKD(x_new)
            x_new_node.parent = x_near_node
            x_new_node.control = u
            x_near_node.children.append(x_new_node)
            self.nodes.append(x_new_node)
            if self.in_goal(x_new):
                print("Goal reached!")
                return True, x_new_node
            if visualize:
                plt.plot([x_near[0], x_new[0]], [x_near[1], x_new[1]], linestyle="--", marker='.', color="black")
        return False, None
    
    def sample_ctrl(self, x_near, x_rand, dt, n = 5):
        controls = []
        for i in range(n):
            c = np.random.rand(2)
            c[0] *= 2
            c[0] -= 1 # [-1, 1]
            c[1] *= np.pi/2 #[-pi/4, pi/4], otherwise the car can move round and round. Not very realistic imho
            c[1] -= np.pi/4
            controls.append(c)
        x_preds = [self.simulate(x_near, u, dt, self.simulate_factor) for u in controls]
        x_preds = [(x_preds[i], controls[i]) for i in range(len(controls))]
        x_new, umin = min(x_preds, key=lambda x: self.metric(x[0], x_rand))
        return x_new, umin



    
    def in_goal(self, loc):
        r = self.goal[-1]
        if self.metric(loc, self.goal[:-1]) <= r**2:
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

    def visualize(self, vtype):
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
            ax.add_patch(matplotlib.patches.Circle(node.location[:2], radius=0.05))
            if node.parent is not None:
                p = node.parent
                diff = node.location - p.location
                ax.add_patch(matplotlib.patches.Arrow(p.location[0], p.location[1], diff[0], diff[1], width=0.1, linestyle="--", linewidth=0.02))





if __name__ == "__main__":            
    parser = ap.ArgumentParser()
    parser.add_argument("--start", nargs="+")
    parser.add_argument("--goal", nargs="+")
    parser.add_argument("--goal_rad")
    parser.add_argument("--map")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--prefix", default=f"rrtkd_{np.random.randint(1000)}")
    args = parser.parse_args()
    
    start = np.array([float(qi) for qi in args.start])
    goal = np.array([float(qi) for qi in args.goal] + [float(args.goal_rad)])
    env = scene_from_file(args.map)
    
    simulate_factor = 10
    rrt = KDRRT(start, goal, env, freebody2D_sample_conf, freebody2D_metric, freebody2D_expand, car_collision, car_collision_path, freebody2D_difference, simulate_analytical, simulate_factor)

    vis_type = "freebody"
    
    visualize_scene(env)
    r = None
    fr=0
    while len(rrt.nodes) < 10000:
        c, r = rrt.step(visualize=True)
        if args.save:
            plt.savefig(f"{args.prefix}_treegrowth_fr{fr:05}.png")
        else:
            plt.pause(0.0001)
        if c:
            print("Goal Reached!")
            break
        print(f"Node count: {len(rrt.nodes)}")
        fr += 1
    
    if not c:
        print("No viable path found!")
        exit(-1)
    #plt.show()
    plt.clf()
    
    path = rrt_get_path(r)
    print(path)
    visualize_path(rrt, vis_type, path)
    if args.save:
        plt.savefig(f"{args.prefix}_path.png")
    plt.show()
    controls = rrt_get_controls(r)
    print(controls)
    animate_car(env, start, controls, save = args.save, prefix=f"{args.prefix}_anim")
