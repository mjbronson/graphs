#!/usr/bin/python3

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# def show_graph(g):
#     nx.draw(g)
#     plt.show()

def graph(num_cliques, clique_sizes, num_extra_connections):
    num_nodes = num_cliques * clique_sizes
    graph = nx.caveman_graph(num_cliques, clique_sizes)

    for _ in range(num_extra_connections):
        a = 1
        b = 1
        while a == b or graph.has_edge(a, b):
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
        graph.add_edge(a, b)
    
    return graph

def matrix(num_cliques, clique_sizes, num_extra_connections):
    return nx.to_numpy_array(graph(num_cliques, clique_sizes, num_extra_connections))

if __name__ == '__main__':
    g = graph(5, 6, 10)
    show_graph(g)
