import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import json
from math import *

class Environment:
    def __init__(self):
        self.obstacles = None
        self.robot = None
        self.limits = [[-10, 10], [-10, 10]]

def rect_to_vertices(r): # Takes in (x, y, theta, w, h), gives (p_c, p_1, p_2, p_3, p_4), in a counter clockwise order
    x, y, theta, w, h = r
    p_c = np.array([x, y])
    p_1 = [x + w/2*cos(theta) - h/2*sin(theta),
           y + w/2*sin(theta) + h/2*cos(theta)]
    p_2 = [x - w/2*cos(theta) - h/2*sin(theta),
           y + h/2*cos(theta) - w/2*sin(theta)]
    p_3 = [x - w/2*cos(theta) + h/2*sin(theta),
           y - w/2*sin(theta) - h/2*cos(theta)]
    p_4 = [x + w/2*cos(theta) + h/2*sin(theta),
           y - h/2*cos(theta) + w/2*sin(theta)]
    p_1 = np.array(p_1)
    p_2 = np.array(p_2)
    p_3 = np.array(p_3)
    p_4 = np.array(p_4)
    return (p_c, p_1, p_2, p_3, p_4)

def generate_environment(number_of_obstacles):
    env = Environment()
    env.obstacles = np.random.rand(number_of_obstacles, 5) # (x, y, theta w, h)
    env.obstacles[:,3:] *= 1.5 # [0, 1.5)
    env.obstacles[:,3:] += 0.5 # [0.5, 2.0)
    env.obstacles[:,2] *= 2*np.pi
    env.obstacles[:,:2] *= 20 # [0, 20)
    env.obstacles[:,:2] -= 10 # [-10, 10)
    return env

def scene_to_file(env, filename):
    env_state = [env.obstacles.tolist(), env.robot, env.limits]
    jstring = json.dumps(env_state)
    print(jstring)
    with open(filename, "w") as f:
        f.write(jstring)

def scene_from_file(filename):
    jstring = ""
    with open(filename, "r") as f:
        jstring = f.read()
    env_dict = json.loads(jstring)
    env = Environment()
    env.obstacles = np.array(env_dict[0])
    env.limits = env_dict[2]
    env.robot = env_dict[1]
    return env

def visualize_scene(env):
    plt.plot()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axis("equal")
    ax = plt.gca()
    for r in env.obstacles:
        rv = rect_to_vertices(r)
        ax.add_patch(matplotlib.patches.Rectangle((rv[3]), r[3], r[4], angle=r[2]*180/np.pi))
    #plt.show()
    pass
def constrain_angle(theta):
    # Theta should always be between 0 and 2pi
    if theta < 0 or theta >= 2*pi:
        r = floor(theta/2/pi)
        theta -= r*2*pi
    return theta
def SAT_rect(r1, r2): # Takes in rectangles in vertex form, return true if colliding, false if  not
    axes = []
    axes.append(r1[2] - r1[1])
    axes.append(r1[3] - r1[2])
    axes.append(r2[2] - r2[1])
    axes.append(r2[3] - r2[1])
    colliding = True
    for axis in axes:
        proj1 = [np.dot(r1[i], axis) for i in range(1, len(r1))]
        proj2 = [np.dot(r2[i], axis) for i in range(1, len(r2))]

        r1_max = np.max(proj1)
        r1_min = np.min(proj1)
        r2_max = np.max(proj2)
        r2_min = np.min(proj2)
        if r1_min > r2_max or r2_min > r1_max: # No collision condition
            return False
    return True # No gap was found, so collision is true


     
# Collision checking
def collision_check(actor, obstacles): # Takes in a rectangular actor and a set of obstacles. Checks for collisions between the actor and all obstacles
    av = rect_to_vertices(actor)
    for v in av:
        if np.any(np.abs(v) > 10): #Out of bounds
            return True
    for o in obstacles:
        ov = rect_to_vertices(o)
        if SAT_rect(av, ov): # True so collision!
            return True
    return False # False so no collision


# For 2D freebody. Configuration is of the form (x,y, theta)

def freebody2D_sample_conf(goal):
    g = goal[:3] # We don't need the radius where we are going
    c = np.random.rand(1)
    if c >= 0.95:
        return g # Return the goal with a 5% chance
    
    conf = np.random.rand(3)
    conf[:2] *= 20 # [0, 20]
    conf[:2] -= 10 # [-10, 10]
    conf[2] *= 2*np.pi #[0, 2*pi]
    return conf

def angle_diff(a1, a2):
    a1 = constrain_angle(a1)
    a2 = constrain_angle(a2)
    diff = a1 - a2
    if diff > np.pi:
        diff = -2*np.pi + a1 - a2
    elif diff < -np.pi:
        diff = 2*np.pi + a1 - a2
    return diff

def freebody2D_difference(q1, q2):
    qdiff = q1 - q2
    theta1 = constrain_angle(q1[2])
    theta2 = constrain_angle(q2[2])
    theta_diff = theta1 - theta2
    if theta_diff > np.pi:
        theta_diff = -2*np.pi + theta1 - theta2
    elif theta_diff < -np.pi:
        theta_diff = 2*np.pi + theta1 - theta2
    return qdiff

def freebody2D_metric(q1, q2):
    # Return squared euclidean distance
    qdiff = freebody2D_difference(q1, q2)
    return np.sum(np.square(qdiff))

def freebody2D_expand(q1, q2, alpha=0.1):
    qdiff = freebody2D_difference(q2, q1)
    qdiff /= np.linalg.norm(qdiff)
    qdiff[2] *= 0.1 # Extra scaling for the angle
    qnew = q1 + qdiff * alpha
    return qnew

def freebody2D_collision(q, env):
    r_actor = (q[0], q[1], q[2], 0.5, 0.3)
    obstacles = env.obstacles
    return collision_check(r_actor, obstacles)

def freebody2D_collision_path(q1, q2, env, n=10): # Use a simple raymarch to see if the path is viable
    qdiff = freebody2D_difference(q2, q1)
    if freebody2D_collision(q1, env) or freebody2D_collision(q2, env):
        return True # If either the start or the end points are in collision, then return true
    for i in range(1, n):
        qp = q1 + qdiff * i/n
        if freebody2D_collision(qp, env):
            return True
    return False # No collisions so far so return false

def arm_sample_conf(goal):
    g = goal[:2]
    c = np.random.rand(1)
    if c >= 0.95:
        return g # Return the goal with a 5% chance
    
    conf = np.random.rand(2)
    conf *= 2*np.pi #[0, 2*pi]
    return conf

def arm_difference(q1, q2):
    qdiff = np.zeros(2)
    qdiff[0] = angle_diff(q1[0], q2[0])
    qdiff[1] = angle_diff(q1[1], q2[1])
    return qdiff

def arm_metric(q1, q2):
    # Return squared euclidean distance
    qdiff = arm_difference(q1, q2)
    metric= np.sum(np.square(qdiff))
    return metric

def arm_expand(q1, q2, alpha=0.1):
    qdiff = arm_difference(q2, q1)
    qdiff /= np.linalg.norm(qdiff)
    qnew = q1 + qdiff * alpha
    qnew2 = np.array([constrain_angle(i) for i in qnew])
    return qnew2

def arm_collision(q, env):
    # Assume length of arm is 1, 
    r_actor = (np.cos(q[0])/2, np.sin(q[0])/2, q[0], 1, 0.001)
    r_actor2 = (np.cos(q[0]) + np.cos(q[0] + q[1])/2,
                np.sin(q[0]) + np.sin(q[0] + q[1])/2,
                q[0] + q[1], 1, 0.001)
    obstacles = env.obstacles
    cond = collision_check(r_actor, obstacles) or collision_check(r_actor2, obstacles)
    return cond

def arm_collision_path(q1, q2, env, n=10): # Use a simple raymarch to see if the path is viable
    qdiff = arm_difference(q2, q1)
    if arm_collision(q1, env) or arm_collision(q2, env):
        return True # If either the start or the end points are in collision, then return true
    for i in range(1, n):
        qp = q1 + qdiff * i/n
        if arm_collision(qp, env):
            return True
    return False # No collisions so far so return false

def animate_arm(env, path, interp=False, n=10, savefig=False, prefix="arm_anim"):
    poses = []
    if interp:
        for i in range(0, len(path) - 1):
            poses.append(path[i])
            diff = arm_difference(path[i + 1], path[i])
            for j in range(n):
                poses.append(path[i] + diff * j/n)
    else:
        poses = path

    for i, pose in enumerate(poses):
        if arm_collision(pose, env):
            print("oh no")
        visualize_scene(env)
        x = [0., np.cos(pose[0]), np.cos(pose[0]) + np.cos(pose[0] + pose[1])]
        y = [0., np.sin(pose[0]), np.sin(pose[0]) + np.sin(pose[0] + pose[1])]
        plt.plot(x, y, "b")
        if savefig:
            plt.savefig(f"{prefix}_fr{i:04}.png")
            plt.clf()
        else:
            plt.pause(0.1)
            plt.clf()

def animate_freebody(env, path, interp=False, n=10, savefig=False, prefix="freebody_anim"):
    poses = []
    if interp:
        for i in range(0, len(path) - 1):
            poses.append(path[i])
            diff = freebody2D_difference(path[i + 1], path[i])
            for j in range(n):
                poses.append(path[i] + diff * j/n)
    else:
        poses = path

    for i, pose in enumerate(poses):
        if freebody2D_collision(pose, env):
            print("oh no")
        visualize_scene(env)
        ax = plt.gca()
        rv = rect_to_vertices([pose[0], pose[1], pose[2], 0.5, 0.3])
        ax.add_patch(matplotlib.patches.Rectangle(rv[3], 0.5, 0.3, angle=pose[2] * 180/np.pi, color="gray"))
        if savefig:
            plt.savefig(f"{prefix}_fr{i:04}.png")
            plt.clf()
        else:
            plt.pause(0.1)
            plt.clf()

def visualize_path_arm(o, p, display_nodes=True):
    o.visualize(vtype="arm")
    ax = plt.gca()
    x = []
    y = []
    ax.add_patch(matplotlib.patches.Circle(p[0], radius=0.1, color=("yellow", 0.5)))
    ax.add_patch(matplotlib.patches.Circle(p[-1], radius=0.1, color=("green", 0.5)))
    if len(p) > 2 and display_nodes:
        for pose in p[1:-1]:
            ax.add_patch(matplotlib.patches.Circle(pose, radius=0.1, color=("orange", 0.5)))
    for pose in p:
        x.append(pose[0])
        y.append(pose[1])
    plt.plot(x, y, color="green")

import time

def get_path_quality(path, metric_fn):
    quality = 0
    prev_pose = path[0]
    for pose in path[1:]:
        quality += np.sqrt(metric_fn(prev_pose, pose))
        # Our metrics are squared, so we must take the square root so that
        # they work in an additive manner. Otherwise, one big chuck will have
        # a much larger distance than the same chunk split into small pieces and
        # added together
        prev_pose = pose
    return 1/quality # 1/dist, too lazy to change the name, output is still a measure of quality

def rrt_get_path(r):
    path = []
    fbpose = r
    while fbpose != None:
        path.append(fbpose.location)
        fbpose = fbpose.parent
    path = path[::-1]
    return path

def simulate_arm(env, iters=10, vis=False):
    ret_dict = {}
    start = arm_sample_conf(np.array([0., 0,]))
    while arm_collision(start, env):
        start =arm_sample_conf(np.array([0., 0.]))
    goal = arm_sample_conf(np.array([1., 1.]))
    while arm_collision(goal, env):
        goal = arm_sample_conf(np.array([1., 1.]))
    g = np.ones(3)
    g[:2] = goal
    g[2] = 2*np.pi/180 # Goal region radius is 2 degrees
    goal = g
    
    successes_rrt = 0
    time_rrt_start = time.time()
    quality_rrt = 0
    for k in range(iters):
        rrt = RRT(start, goal, env, arm_sample_conf, arm_metric, arm_expand, arm_collision, arm_collision_path, arm_difference)
        r = None
        for i in range(500):
            c, r = rrt.step()
            if c:
                successes_rrt += 1
                break
        path = rrt_get_path(r)
        quality_rrt += get_path_quality(path, arm_metric)
        if vis:
            print("RRT Vis")
            visualize_path_arm(rrt, path, False)
            plt.show()
    time_rrt = time.time() - time_rrt_start
    time_rrt /= iters
    quality_rrt /= successes_rrt
    print(f"RRT Quality: {quality_rrt}")

    successes_prm = 0
    quality_prm = 0
    time_prm_start = time.time()
    for k in range(iters):
        prm = PRM(start, goal, env, arm_sample_conf, arm_metric, arm_expand,
          arm_collision, arm_collision_path, arm_difference, 2*np.pi/10)
        for i in range(500):
            prm.step()
            if i == 499:
                path = prm.a_star()
                if len(path) != 0:
                    quality_prm += get_path_quality(path, arm_metric)
                    successes_prm += 1
                    break
        path = prm.a_star()
        if len(path) > 0 and vis:
            visualize_path_arm(prm, path)
            plt.show()
    time_prm = time.time() - time_prm_start
    time_prm /= iters
    quality_prm /= successes_prm
    print(f"PRM Quality: {quality_prm}")

    successes_ao = 0
    quality_ao = 0
    time_ao_start = time.time()
    for k in range(iters):
        prm_ao = PRM(start, goal, env, arm_sample_conf, arm_metric, arm_expand,
          arm_collision, arm_collision_path, arm_difference, 1.0, True)
        for i in range(500):
            prm_ao.step()
            if i == 499:
                path = prm_ao.a_star()
                if len(path) != 0:
                    quality_ao += get_path_quality(path, arm_metric)
                    successes_ao += 1
                    break
        path = prm_ao.a_star()
        if len(path) > 0 and vis:
            visualize_path_arm(prm_ao, path)
            plt.show()
    time_ao = time.time() - time_ao_start
    time_ao /= iters
    quality_ao /= successes_ao
    print(f"PRM AO Quality: {quality_ao}")

    ret_dict["rrt"] = (successes_rrt, time_rrt, quality_rrt)
    ret_dict["prm"] = (successes_prm, time_prm, quality_prm)
    ret_dict["ao"] = (successes_ao, time_ao, quality_ao)
    return ret_dict

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

def animate_car(env, start, controls, dt=0.1, save=False,prefix="car_anim", simulate_factor=10):
    path = [start]
    x_curr = start
    for u in controls:
        for i in range(simulate_factor):
            x_curr = simulate(x_curr, u, dt, simulate_factor=1) # Interpolation of sorts
            path.append(x_curr)
    for i, pose in enumerate(path):
        if car_collision(pose, env):
            print("oh no")
        visualize_scene(env)
        ax = plt.gca()
        rv = rect_to_vertices([pose[0], pose[1], pose[2], 0.8, 0.5])
        ax.add_patch(matplotlib.patches.Rectangle(rv[3], 0.8, 0.5, angle=pose[2] * 180/np.pi, color="gray"))
        if save:
            plt.savefig(f"{prefix}_fr{i:03}.png")
        plt.show()

def visualize_arm_conf(conf, color="gray"):
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    x = [0., np.cos(conf[0]), np.cos(conf[0]) + np.cos(conf[0] + conf[1])]
    y = [0., np.sin(conf[0]), np.sin(conf[0]) + np.sin(conf[0] + conf[1])]
    plt.plot(x, y, marker=".", color=color)

def visualize_freebody_conf(conf, color="gray"):
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    ax = plt.gca()
    rv = rect_to_vertices([conf[0], conf[1], conf[2], 0.5, 0.3])
    ax.add_patch(matplotlib.patches.Rectangle(rv[3], 0.5, 0.3, angle=conf[2] * 180/np.pi, color=color))

def visualize_path(o,vis_type, p, display_nodes=True):
    o.visualize(vtype=vis_type)
    ax = plt.gca()
    x = []
    y = []
    ax.add_patch(matplotlib.patches.Circle(p[0], radius=0.1, color=("yellow", 0.5)))
    ax.add_patch(matplotlib.patches.Circle(p[-1], radius=0.1, color=("green", 0.5)))
    if len(p) > 2 and display_nodes:
        for pose in p[1:-1]:
            ax.add_patch(matplotlib.patches.Circle(pose, radius=0.1, color=("orange", 0.5)))
    for pose in p:
        x.append(pose[0])
        y.append(pose[1])
    plt.plot(x, y, color="green")


def get_path_quality(path, metric_fn):
    quality = 0
    prev_pose = path[0]
    for pose in path[1:]:
        quality += np.sqrt(metric_fn(prev_pose, pose))
        # Our metrics are squared, so we must take the square root so that
        # they work in an additive manner. Otherwise, one big chuck will have
        # a much larger distance than the same chunk split into small pieces and
        # added together
        prev_pose = pose
    return 1/quality # 1/dist, too lazy to change the name, output is still a measure of quality

def rrt_get_path(r):
    path = []
    fbpose = r
    while fbpose != None:
        path.append(fbpose.location)
        fbpose = fbpose.parent
    path = path[::-1]
    return path

def simulate_arm(env, iters=10, vis=False):
    ret_dict = {}
    start = arm_sample_conf(np.array([0., 0,]))
    while arm_collision(start, env):
        start =arm_sample_conf(np.array([0., 0.]))
    goal = arm_sample_conf(np.array([1., 1.]))
    while arm_collision(goal, env):
        goal = arm_sample_conf(np.array([1., 1.]))
    g = np.ones(3)
    g[:2] = goal
    g[2] = 2*np.pi/180 # Goal region radius is 2 degrees
    goal = g
    
    successes_rrt = 0
    time_rrt_start = time.time()
    quality_rrt = 0
    for k in range(iters):
        rrt = RRT(start, goal, env, arm_sample_conf, arm_metric, arm_expand, arm_collision, arm_collision_path, arm_difference)
        r = None
        for i in range(500):
            c, r = rrt.step()
            if c:
                successes_rrt += 1
                break
        path = rrt_get_path(r)
        quality_rrt += get_path_quality(path, arm_metric)
        if vis:
            print("RRT Vis")
            visualize_path_arm(rrt, path, False)
            plt.show()
    time_rrt = time.time() - time_rrt_start
    time_rrt /= iters
    quality_rrt /= successes_rrt
    print(f"RRT Quality: {quality_rrt}")

    successes_prm = 0
    quality_prm = 0
    time_prm_start = time.time()
    for k in range(iters):
        prm = PRM(start, goal, env, arm_sample_conf, arm_metric, arm_expand,
          arm_collision, arm_collision_path, arm_difference, 2*np.pi/10)
        for i in range(500):
            prm.step()
            if i == 499:
                path = prm.a_star()
                if len(path) != 0:
                    quality_prm += get_path_quality(path, arm_metric)
                    successes_prm += 1
                    break
        path = prm.a_star()
        if len(path) > 0 and vis:
            visualize_path_arm(prm, path)
            plt.show()
    time_prm = time.time() - time_prm_start
    time_prm /= iters
    quality_prm /= successes_prm
    print(f"PRM Quality: {quality_prm}")

    successes_ao = 0
    quality_ao = 0
    time_ao_start = time.time()
    for k in range(iters):
        prm_ao = PRM(start, goal, env, arm_sample_conf, arm_metric, arm_expand,
          arm_collision, arm_collision_path, arm_difference, 1.0, True)
        for i in range(500):
            prm_ao.step()
            if i == 499:
                path = prm_ao.a_star()
                if len(path) != 0:
                    quality_ao += get_path_quality(path, arm_metric)
                    successes_ao += 1
                    break
        path = prm_ao.a_star()
        if len(path) > 0 and vis:
            visualize_path_arm(prm_ao, path)
            plt.show()
    time_ao = time.time() - time_ao_start
    time_ao /= iters
    quality_ao /= successes_ao
    print(f"PRM AO Quality: {quality_ao}")

    ret_dict["rrt"] = (successes_rrt, time_rrt, quality_rrt)
    ret_dict["prm"] = (successes_prm, time_prm, quality_prm)
    ret_dict["ao"] = (successes_ao, time_ao, quality_ao)
    return ret_dict

def visualize_arm_cspace(env, n=100):
    x = np.linspace(0, 2*np.pi, n)
    y = np.linspace(0, 2*np.pi, n)
    plt.plot()
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 2*np.pi)
    plt.axis("equal")
    ax = plt.gca()
    for j in y:
        for i in x:
            if arm_collision(np.array([i, j]), env):
                ax.add_patch(matplotlib.patches.Circle((i, j), radius=0.02, color="red"))
matplotlib.use("TkAgg")
