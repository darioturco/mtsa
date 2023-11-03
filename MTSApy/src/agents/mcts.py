import torch
import torch.nn as nn
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
from src.agents.agent import Agent
import networkx as nx


class Node(object):
    def __init__(self, fv):
        self.fv = fv
        self.n = 0
        self.w = 0
        self._to = []
        self._from = []
        self.l = 0
        self.win = 0    # cantidad de veces que fue un estrado ganador
        # winrate = Cantidad de veces que este es un estado final dividido la cantidad de veces visitado

    def update(self, reward, to, l):
        self.n += 1
        self.w += reward
        self.l += l
        if to is not None:
            idx = self.get_index_of_transition(to)
            if idx == -1:
                self._to.append(Edge(self.fv, to.fv))
            else:
                self._to[idx].inc_weight()

    def get_index_of_transition(self, to):
        for i, edge in enumerate(self._to):
            if edge._to == to.fv:
                return i

        return -1


    def __eq__(self, other):
        return self.fv == other.fv

    def add_from(self, node):
        if node not in self._from:
            self._from.append(node)

    def win_rate(self):
        if self.n >= 1:
            return self.win / self.n
            #return self.l * self.win / self.n
        else:
            return 0




    #def __eq__(self, other):
    #    return self.fv == other.fv
    def bfs(self):
        res = []
        visited = [self]
        queue = [self]
        distances = [0]
        while len(queue) > 0:
            n = queue.pop(0)
            d = distances.pop(0)

            # Do what I need with n
            res.append((n, d))

            for edge in n._to:
                node_fv = edge._from
                print(node_fv)
                if node_fv not in visited:
                    queue.append(node_fv)
                    visited.append(node_fv)
                    distances.append(d+1)

        return res


    def __repr__(self):
        #return f"({self.n}, {self.w})"
        return f"({self.fv})"

class Edge(object):
    def __init__(self, f, t):
        self._from = f
        self._to = t
        self.weight = 1

    def inc_weight(self):
        self.weight += 1

class MCTS(Agent):
    def __init__(self, env):

        self.env = env
        self.eval = {}  # Dictionary of {fv: (n, w)}
        self.trained = False
        self.c = 1.0

    def train(self, max_eps=100):

        for eps in range(max_eps):
            #print(f"Epsode: {eps}")
            obs = self.env.reset()
            done = False
            rewards = []
            trace = []

            while not done:
                action = self.get_action(obs, 1.0, c=self.c)   # aca deberia pasarle el self.c
                obs2, reward, done, info = self.env.step(action)

                fv = self.feature_vector_to_number(obs[action])
                rewards.append(reward)
                trace.append(fv)
                obs = obs2

            print(f"Epsode: {eps} - {trace}")
            acumulated_reward = sum(rewards)
            self.update_eval(trace, acumulated_reward)


        self.G = nx.DiGraph()
        node_colors = []
        for fv, node in self.eval.items():
            a = hex(int(np.clip(node.win * 100, 0, 255)))
            if len(a) == 3:
                node_colors.append(f"#AA000{str(a)[2:]}")
            else:
                node_colors.append(f"#AA00{str(a)[2:]}")

            self.G.add_node(fv)

        for fv, node in self.eval.items():
            for n in node._to:
                #self.G.add_edge(fv, n.fv, weight=1)
                self.G.add_edge(n._from, n._to, weight=1/n.weight)


        s = 16
        plt.figure(1, figsize=(s, s))
        nx.draw(self.G, pos=nx.circular_layout(self.G), with_labels=True, node_size=3000, node_color=node_colors)
        #plt.show()

        self.trained = True

    def update_eval(self, trace, acumulated_reward):
        for idx_fv in range(len(trace)-1):
            fv = trace[idx_fv]
            fv_next = trace[idx_fv+1]
            if fv not in self.eval:
                self.eval[fv] = Node(fv)

            if fv_next not in self.eval:
                self.eval[fv_next] = Node(fv_next)

            self.eval[fv].update(acumulated_reward, self.eval[fv_next], len(trace)-idx_fv)
            #self.eval[fv_next].add_from(self.eval[fv])

        last_fv = trace[len(trace)-1]
        self.eval[last_fv].update(acumulated_reward, None, len(trace))
        self.eval[last_fv].win += 1

    def path_weight(self, ds):
        return nx.path_weight(self.G, ds, 'weight')

    #def self.vecinos_a(self, k, node):
    #    res = [node]
    #    for kk in range(k):


    def getUCB(self, fv):
        if fv not in self.eval:
            return 0

        k = 3
        node = self.eval[fv]

        lista = node.bfs()
        print("holaa")
        print(lista)


         #return max([self.vecinos_a(kk, node) ])



        #node = self.eval[fv]
        return sum([ self.eval[t._from].win_rate() * t.weight / node.n for t in node._to])


        """
        # If only the target is specified, return a dictionary keyed by sources with a list of nodes in a shortest path from one of the sources to the target.
        distances = nx.shortest_path(self.G, source=fv, weight='weight', method='dijkstra')
        distances = {n_fv: self.path_weight(path) for n_fv, path in distances.items()}
        nodes_values = [self.eval[n_fv].win_rate() / d if d > 0 else 0.0 for n_fv, d in distances.items()]
        return max(nodes_values)
        """






    def get_action(self, state, *args, **kwargs):
        epsilon = args[0]
        if np.random.rand() <= epsilon:
            return np.random.randint(len(state))
        else:
            state_fv = [self.feature_vector_to_number(s) for s in state]
            UCBs = []
            for fv in state_fv:
                UCBs.append(self.getUCB(fv))
            return np.argmax(UCBs)

            #max_value = np.max(UCBs)
            #idx_maxUCBs = [i for i, ucb in enumerate(UCBs) if ucb == max_value]
            #return random.choice(idx_maxUCBs)
