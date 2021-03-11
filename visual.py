#!/usr/bin/python3

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gc


def seir_nodes(g, matrix, seir, layout):
    susceptible = np.where(seir[:, 0] > 0)[0]
    exposed = np.where(seir[:, 1] > 0)[0]
    infectious = np.where(seir[:, 2] > 0)[0]
    recovered = np.where(seir[:, 3] > 0)[0]
    nx.draw_networkx_nodes(g, pos=layout, nodelist=susceptible, node_color='green')
    nx.draw_networkx_nodes(g, pos=layout, nodelist=exposed, node_color='yellow')
    nx.draw_networkx_nodes(g, pos=layout, nodelist=infectious, node_color='red')
    nx.draw_networkx_nodes(g, pos=layout, nodelist=recovered, node_color='gray')
    
def sei_nodes(g, matrix, sei, layout):
    susceptible = np.where(sei[:, 0] > 0)[0]
    exposed = np.where(sei[:, 1] > 0)[0]
    infectious = np.where(sei[:, 2] > 0)[0]
    nx.draw_networkx_nodes(g, pos=layout, nodelist=susceptible, node_color='green')
    nx.draw_networkx_nodes(g, pos=layout, nodelist=exposed, node_color='yellow')
    nx.draw_networkx_nodes(g, pos=layout, nodelist=infectious, node_color='red')

def seir_edges(g, matrix, seir, layout):
    newly_exposed = np.where(seir[:, 1] == 1)[0]
    edges = []
    for x in newly_exposed:
        near_newly_exposed = matrix[x] > 0
        if np.count_nonzero(near_newly_exposed):
            infectious = seir[:, 2] > 0
            infectious_near_newly_exposed = np.where(infectious & near_newly_exposed)[0]
            if len(infectious_near_newly_exposed) == 0:
                recovered = seir[:, 3] > 0
                infectious_near_newly_exposed = np.where(recovered & near_newly_exposed)[0]
            if len(infectious_near_newly_exposed):
                edge = (x, infectious_near_newly_exposed[0])
                edges.append(edge)
    nx.draw_networkx_edges(g, pos=layout)
    nx.draw_networkx_edges(g, pos=layout, edgelist=edges, edge_color='green', width=7)
    
def sei_edges(g, matrix, sei, layout):
    newly_exposed = np.where(sei[:, 1] == 1)[0]
    edges = []
    for x in newly_exposed:
        near_newly_exposed = matrix[x] > 0
        if np.count_nonzero(near_newly_exposed):
            infectious = sei[:, 2] > 0
            infectious_near_newly_exposed = np.where(infectious & near_newly_exposed)[0]
            if len(infectious_near_newly_exposed):
                edge = (x, infectious_near_newly_exposed[0])
                edges.append(edge)
    nx.draw_networkx_edges(g, pos=layout)
    nx.draw_networkx_edges(g, pos=layout, edgelist=edges, edge_color='green', width=7)

def visualize(m, sim_info1, n1=None, e1=None, sim_info2=None, n2=None, e2=None, layout=None, time_delay=.5):
    
    g = nx.from_numpy_array(m)
    if layout is None:
        layout = nx.spring_layout(g)
    if sim_info2 is None:
        sim_info2 = sim_info1

    for time_step in range(min(len(sim_info1), len(sim_info2))):
        s1 = sim_info1[min(time_step, len(sim_info1) - 1)]
        s2 = sim_info2[min(time_step, len(sim_info2) - 1)]
    
        # Draw Nodes
        if n1 is None and n2 is None:
            nx.draw_networkx_nodes(g, pos=layout)
        elif n1 is not None:
            n1(g, m, s1, layout)
        else:
            n2(g, m, s2, layout)

        # Draw Edges
        if e1 is None and e2 is None:
            nx.draw_networkx_edges(g, pos=layout)
        elif e1 is not None:
            e1(g, m, s1, layout)
        else:
            e2(g, m, s2, layout)

        nx.draw_networkx_labels(g, pos=layout)
        plt.title('Time Step: {ts}'.format(ts=time_step))
        plt.pause(time_delay)
        plt.clf()
    

def visualize_seirs(matrix, seirs, time_delay=1):
    g = nx.from_numpy_matrix(matrix)
    layout = nx.spring_layout(g)
    for time_step, seir in enumerate(seirs):
        seir_nodes(g, matrix, seir, layout)
        nx.draw_networkx_labels(g, pos=layout)
        seir_edges(g, matrix, seir, layout)
        plt.title('Time Step: {ts}'.format(ts=time_step))
        plt.pause(time_delay)
        plt.clf()

def visualize_seis(matrix, seis, time_delay=1):
    g = nx.from_numpy_matrix(matrix)
    layout = nx.spring_layout(g)
    for time_step, sei in enumerate(seis):
        gc.collect()
        susceptible = np.where(sei[:, 0] > 0)[0]
        exposed = np.where(sei[:, 1] > 0)[0]
        infectious = np.where(sei[:, 2] > 0)[0]
        nx.draw_networkx_nodes(g, pos=layout, nodelist=susceptible, node_color='green')
        nx.draw_networkx_nodes(g, pos=layout, nodelist=exposed, node_color='yellow')
        nx.draw_networkx_nodes(g, pos=layout, nodelist=infectious, node_color='red')
        nx.draw_networkx_labels(g, pos=layout)
        nx.draw_networkx_edges(g, pos=layout)
        plt.title('Time Step: {ts}'.format(ts=time_step))
        plt.pause(time_delay)


if __name__ == '__main__':
    matrix = np.array([[0, 1], [1, 0]])
    seirs = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        [[2, 0, 0, 0], [0, 2, 0, 0]],
        [[0, 1, 0, 0], [0, 0, 1, 0]],
        [[0, 2, 0, 0], [0, 0, 2, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1]],
        [[0, 0, 2, 0], [0, 0, 0, 2]],
        [[0, 0, 0, 1], [1, 0, 0, 0]],
        [[0, 0, 0, 2], [2, 0, 0, 0]],

        [[1, 0, 0, 0], [0, 1, 0, 0]],
        [[2, 0, 0, 0], [0, 2, 0, 0]],
        [[0, 1, 0, 0], [0, 0, 1, 0]],
        [[0, 2, 0, 0], [0, 0, 2, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1]],
        [[0, 0, 2, 0], [0, 0, 0, 2]],
        [[0, 0, 0, 1], [1, 0, 0, 0]],
        [[0, 0, 0, 2], [2, 0, 0, 0]],
        ])

    seis = np.array([
        [[1, 0, 0], [0, 1, 0]],
        [[2, 0, 0], [0, 2, 0]],
        [[0, 1, 0], [0, 0, 1]],
        [[0, 2, 0], [0, 0, 2]],
        [[0, 0, 1], [1, 0, 0]],
        [[0, 0, 2], [2, 0, 0]],
        [[1, 0, 0], [0, 1, 0]],
        [[2, 0, 0], [0, 2, 0]],

        [[0, 1, 0], [0, 0, 1]],
        [[0, 2, 0], [0, 0, 2]],
        [[0, 0, 1], [1, 0, 0]],
        [[0, 0, 2], [2, 0, 0]],
        [[1, 0, 0], [0, 1, 0]],
        [[2, 0, 0], [0, 2, 0]],
        [[0, 1, 0], [0, 0, 1]],
        [[0, 2, 0], [0, 0, 2]],
        ])

    visualize(matrix, sim_info1=seis, e1=sei_edges, sim_info2=seirs, n2=seir_nodes, time_delay=.5)
