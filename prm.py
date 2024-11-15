from utils import *
import heapq

import argparse as ap
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import csv



class GraphRM:
    def __init__(self):
        self.graph_dict = {}
    def add_node(self, conf):
        conf = tuple(conf)
        self.graph_dict[conf] = set()
    def add_edge(self, conf1, conf2):
        conf1 = tuple(conf1)
        conf2 = tuple(conf2)
        self.graph_dict[conf1].add(conf2)
        self.graph_dict[conf2].add(conf1)
    def __getitem__(self, key):
        key = tuple(key)
        nodes = self.graph_dict[key]
        return [np.array(node) for node in nodes]
    
class IndexAttributes:
    def __init__(self, ind):
        self.ind = ind # Tuple
        self.f = inf
        self.g = inf
        self.h = inf
        self.parent = None
    def __lt__(self, other):
        return self.f < other.f
    def __eq__(self, other):
        return self.f == other.f
    def __gt__(self, other):
        return self.f > other.f
    def __repr__(self):
        return f"({self.ind}, f={self.f})"
    def set_parent(self,indattr):
        self.parent = indattr

class PRM:
    def __init__(self, start, goal_region, env, sample_conf, metric, expand, collision, collision_path, diff_func, radius, ao=False):
        self.start = start
        self.goal = np.array(goal_region) # (q_goal, radius)
        self.roadmap = GraphRM()
        self.roadmap.add_node(start)
        self.roadmap.add_node(goal_region[:-1])
        self.nodes = [start, goal_region[:-1]]
        self.sample_conf = sample_conf
        self.metric = metric
        self.expand = expand
        self.collision = collision
        self.collision_path = collision_path
        self.diff_func = diff_func
        self.env = env
        self.raymarch_scale = 1
        self.ao = ao
        self.radius = radius
    
    def step(self):
        x_rand = self.sample_conf(self.goal)
        if not self.collision(x_rand, self.env):
            self.roadmap.add_node(x_rand)
            self.nodes.append(x_rand)
            x_nears = None
            if self.ao:
                x_nears = self.knn_ao(x_rand) # When k > 2e * log(n), PRM is AO
            else:
                x_nears = self.knn(x_rand)
            for x_near in x_nears:
                dist = np.sqrt(self.metric(x_rand, x_near)) # Use a square root because all the metrics we use are squared
                if not self.collision_path(x_rand, x_near, self.env, n=ceil(dist/0.1)*self.raymarch_scale):
                    self.roadmap.add_edge(x_rand, x_near)

    
    def rnn(self, x_rand, r=3):
        nearest = []
        for node in self.nodes:
            if self.metric(x_rand, node) <= r**2:
                nearest.append(node)
        return nearest
    
    def knn_ao(self, x_rand, k_prm=2*np.e):
        k = k_prm * log(len(self.nodes))
        sorted_nodes = sorted(self.nodes, key=lambda x: self.metric(x_rand, x))
        return sorted_nodes[:ceil(k)]

    def knn(self, x_rand, k=6):
        sorted_nodes = sorted(self.nodes, key=lambda x: self.metric(x_rand, x))
        return sorted_nodes[:ceil(k)]

    def a_star(self):
        end = self.goal[:-1]
        attr_map = {}
        visited = set()
        h = []
        start_attr = IndexAttributes(self.start)
        start_attr.f = 0
        start_attr.g = 0
        heapq.heappush(h, start_attr)
        while len(h) != 0:
            ind_attr = heapq.heappop(h)
            ind = ind_attr.ind
            if np.all(ind == end):
                print("Found destination!")
                path = []
                attr = ind_attr
                while True: #Backtrack
                    path.append(attr.ind)
                    if attr.parent is None:
                        break
                    attr = attr.parent
                return path[::-1]
            ns = self.roadmap[ind]
            for n in ns:
                if tuple(n) in visited:
                    #print("Already Visited, Skipping..")
                    continue # Skip since we have already closed the node

                new_iattr = IndexAttributes(n)
                new_iattr.g = ind_attr.g + np.sqrt(self.metric(ind, n))
                new_iattr.h = np.sqrt(self.metric(n, end))
                new_iattr.f = new_iattr.g + new_iattr.h
                new_iattr.parent = ind_attr

                if tuple(n) in attr_map and attr_map[tuple(n)].f <= new_iattr.f:
                    #print("Have a better path, skipping..")
                    continue #skip if we already have a better path to thise node
                else:
                    attr_map[tuple(n)] = new_iattr
                    heapq.heappush(h, new_iattr)
            visited.add(tuple(ind))
        return []
    def visualize(self, vtype="freebody"):
        if vtype == "freebody":
            visualize_scene(self.env)
            ax = plt.gca()
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            drawn_set = set()
            for conf in self.nodes:
                ax.add_patch(matplotlib.patches.Circle(conf[:2], radius=0.08))
                drawn_set.add(tuple(conf))
                for n in self.roadmap.graph_dict[tuple(conf)]:
                    if n in drawn_set:
                        continue
                    diff = n[:2] - conf[:2]
                    ax.add_patch(matplotlib.patches.Arrow(conf[0], conf[1], diff[0], diff[1], width=0.02, linewidth=0.02, linestyle="--"))
        elif vtype == "arm":
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
                        print("Collision!")
                        ax.add_patch(matplotlib.patches.Circle((i, j), radius=0.02, color="red"))
            
            drawn_set = set()
            for conf in self.nodes:
                ax.add_patch(matplotlib.patches.Circle(conf, radius=0.08))
                drawn_set.add(tuple(conf))
                for n in self.roadmap.graph_dict[tuple(conf)]:
                    # if n in drawn_set:
                    #     continue
                    diff = self.diff_func(n, conf)
                    x = []
                    y = []
                    for i in range(11):
                        x.append(conf[0] + diff[0] * i/10)
                        y.append(conf[1] + diff[1]*i/10)
                    plt.plot(x, y, linewidth=0.2, linestyle="--", color="blue")
            pass


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--robot")
    parser.add_argument("--start", nargs="+")
    parser.add_argument("--goal", nargs="+")
    parser.add_argument("--map")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--prefix", required=False, default=f"prm_default{np.random.randint(1000)}")
    args = parser.parse_args()
    
    start = np.array([float(qi) for qi in args.start])
    goal = np.array([float(qi) for qi in args.goal] + [0.3])
    env = scene_from_file(args.map)
    prm = None
    vis_type = None
    if args.robot == "arm":
        prm = PRM(start, goal, env, arm_sample_conf, arm_metric,
                  arm_expand, arm_collision, arm_collision_path, arm_difference, 2*np.pi/10)
        vis_type = "arm"
    else:
        prm = PRM(start, goal, env, freebody2D_sample_conf, freebody2D_metric, freebody2D_expand,
              freebody2D_collision, freebody2D_collision_path, freebody2D_difference, 2.0)
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
