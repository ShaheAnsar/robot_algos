from utils import *
from rrt import TreeNodeKD

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
        x_new, u = self.sample_ctrl(x_near, x_rand, dt) # Since we have a very simple 2D robot, we just use the displacement vector to create our accel vector

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
                plt.plot([x_near[0], x_new[0]], [x_near[1], x_new[1]], linestyle="--", marker='.', color="orange")
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
    def visualize(self):
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
