#!/usr/bin/python3

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import sim_helper as sh

def simulate(m, seir, num_steps, days_exposed, days_infectious, s_to_e):
    """
    m: adjacency matrix of graph
    seir: starting numbers of s, e, i, r
    num_steps: number of steps to run for
    """
    seir = np.array(seir)
    seirs = np.zeros((num_steps, *seir.shape))
    days_exposed = days_exposed
    days_infectious = days_infectious
    for step in range(num_steps):
        # Probabilistic vales to use during the simulation
        probs = np.random.rand(len(m))
        # Infectious to Recoved
        to_r_filter = seir[:, 2] > days_infectious
        seir[to_r_filter, 3] = -1
        seir[to_r_filter, 2] = 0
        # Exposed to Infectious
        to_i_filter = seir[:, 1] > days_exposed
        seir[to_i_filter, 2] = -1
        seir[to_i_filter, 1] = 0
        # Susceptible to Exposed
        i_filter = seir[:, 2] > 0
        # Getting infectious rates for each infectious person
        to_e_probs = 1 - (np.prod((1 - (m * s_to_e))[:, i_filter], axis=1))
        to_e_filter = (seir[:, 0] > 0) & (probs < to_e_probs)
        seir[to_e_filter, 1] = -1
        seir[to_e_filter, 0] = 0
        # Tracking days and seirs
        seir[(seir > 0)] += 1
        seir[(seir < 0)] = 1
        seirs[step] = seir
        if np.sum(seir[:, 1] > 0) + np.sum(seir[:, 2] > 0) == 0:
            return seirs[:step]
    return seirs

def quick_sim(m, num_steps=100, days_exposed=3, days_infectious=5, s_to_e=0.3):
    return simulate(m, sh.get_seir(len(m), 1), 500, 5, 5, 0.3)


if __name__ == '__main__':
    import sim_helper as sh
    def small_world_caveman(num_cliques, clique_sizes, num_extra_connections):
        num_nodes = num_cliques * clique_sizes
        graph = nx.caveman_graph(num_cliques, clique_sizes)

        for _ in range(num_extra_connections):
            a = 1
            b = 1
            while a == b or graph.has_edge(a, b):
                a = random.randint(0, num_nodes - 1)
                b = random.randint(0, num_nodes - 1)
            graph.add_edge(a, b)
        
        return nx.to_numpy_array(graph)

    g = small_world_caveman(5, 6, 10)
    # print('graph shape', g)
    # sh.display_graph_from_numpy(g)
    seirs = simulate(g, sh.get_seir(len(g), 1), 500, 5, 5, 0.3)
    sh.print_seir_stats(seirs)

    import visual
    visual.visualize_seirs(g, seirs, .1)
