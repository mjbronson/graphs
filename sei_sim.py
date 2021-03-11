#!/usr/bin/python3

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import sim_helper as sh

def simulate(m, sei, num_steps, days_exposed, days_infectious, s_to_e):
    """
    m: adjacency matrix of graph
    sei: starting numbers of s, e, i, r
    num_steps: number of steps to run for
    """
    sei = np.array(sei)
    seis = np.zeros((num_steps, *sei.shape))
    days_exposed = days_exposed
    days_infectious = days_infectious
    for step in range(num_steps):
        # Probabilistic vales to use during the simulation
        probs = np.random.rand(len(m))
        # Infectious to Recoved
        to_r_filter = sei[:, 2] > days_infectious
        sei[to_r_filter, 0] = -1
        sei[to_r_filter, 2] = 0
        # Exposed to Infectious
        to_i_filter = sei[:, 1] > days_exposed
        sei[to_i_filter, 2] = -1
        sei[to_i_filter, 1] = 0
        # Susceptible to Exposed
        i_filter = sei[:, 2] > 0
        to_e_probs = 1 - (np.prod((1 - (m * s_to_e))[:, i_filter], axis=1))
        to_e_filter = (sei[:, 0] > 0) & (probs < to_e_probs)
        sei[to_e_filter, 1] = -1
        sei[to_e_filter, 0] = 0
        # Tracking days and seis
        sei[(sei > 0)] += 1
        sei[(sei < 0)] = 1
        seis[step] = sei
        if np.sum(sei[:, 1] > 0) + np.sum(sei[:, 2] > 0) == 0:
            return seis[:step]
    return seis


def quick_sim(m, num_steps=100, days_exposed=3, days_infectious=5, s_to_e=0.3):
    return simulate(m, sh.get_sei(len(m), 1), 500, 5, 5, 0.3)


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
        
        return nx.to_numpy_matrix(graph)

    g = small_world_caveman(5, 6, 10)
    # sh.display_graph_from_numpy(g)
    seis = simulate(g, sh.get_sei(len(g), 1), 500, 5, 5, 0.3)
    sh.print_seis_stats(seis)
