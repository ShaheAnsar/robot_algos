from utils import *
from rrt import RRT
from prm import PRM
from multiprocessing import Pool

def compare_run(env, iters=10, vis=False):
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

def comparison_run(env, iters=10, t="freeBody", vis=False):
    ret_dict = {}
    sample_fn = None
    collision_fn = None
    metric_fn = None
    expand_fn = None
    collision_path_fn = None
    difference_fn = None
    if t == "arm":
        sample_fn = arm_sample_conf
        collision_fn = arm_collision
        metric_fn = arm_metric
        expand_fn = arm_expand
        collision_path_fn = arm_collision_path
        difference_fn = arm_difference
    else:
        sample_fn = freebody2D_sample_conf
        collision_fn = freebody2D_collision
        metric_fn = freebody2D_metric
        expand_fn = freebody2D_metric
        collision_path_fn = freebody2D_collision_path
        difference_fn = freebody2D_difference

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
        if c:
            path = rrt_get_path(r)
            quality_rrt += get_path_quality(path, arm_metric)
            if vis:
                print("RRT Vis")
                visualize_path_arm(rrt, path, False)
                plt.show()
    time_rrt = time.time() - time_rrt_start
    time_rrt /= iters
    if successes_rrt > 0:
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
    if successes_prm > 0:
        quality_prm /= successes_prm
    print(f"PRM Quality: {quality_prm}")

    successes_ao = 0
    quality_ao = 0
    time_ao_start = time.time()
    for k in range(iters):
        prm_ao = PRM(start, goal, env, arm_sample_conf, arm_metric, arm_expand,
          arm_collision, arm_collision_path, arm_difference, 1.0, ao=True)
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
    if successes_ao > 0:
        quality_ao /= successes_ao
    print(f"PRM AO Quality: {quality_ao}")

    ret_dict["rrt"] = (successes_rrt, time_rrt, quality_rrt)
    ret_dict["prm"] = (successes_prm, time_prm, quality_prm)
    ret_dict["ao"] = (successes_ao, time_ao, quality_ao)
    return ret_dict

envs = [ scene_from_file(f"env_{i}.json") for i in range(5)]
#r = compare_run(envs[0], iters=1)
#print(r)
with Pool(5) as p:
    ret = p.map(comparison_run, envs)
    print(ret)

