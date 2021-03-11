#!/usr/bin/python3

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

def display_graph(g):
    nx.draw(g)
    plt.show()

def display_graph_from_numpy(matrix, layout=None):
    g = nx.from_numpy_matrix(matrix)
    nx.draw(g, pos=layout, node_size=200)
    plt.show()

def get_matrix_from_graph(g):
    return np.array(nx.to_numpy_matrix(g), dtype=np.float64)

def get_seir(num_nodes, num_infectious):
    seir = np.zeros((num_nodes, 4))
    seir[:, 0] = 1
    for _ in range(num_infectious):
        random_person = random.randint(0, num_nodes - 1)
        while seir[random_person, 2] > 0:
            random_person = random.randint(0, num_nodes - 1)
        seir[random_person, 0] = 0
        seir[random_person, 2] = 1
    return seir

def get_sei(num_nodes, num_infectious):
    sei = np.zeros((num_nodes, 3))
    sei[:, 0] = 1
    for _ in range(num_infectious):
        random_person = random.randint(0, num_nodes - 1)
        while sei[random_person, 2] > 0:
            random_person = random.randint(0, num_nodes - 1)
        sei[random_person, 0] = 0
        sei[random_person, 2] = 1
    return sei

def seirs_exposed(seirs):
    return np.count_nonzero(seirs[:, :, 1] == 1) + 1

def seis_exposed(seis):
    return np.count_nonzero(seis[:, :, 1] == 1) + 1


def print_seir_stats(seir):
    print('Num S:', np.sum(seir[0]>0))
    print('Num E', np.sum(seir[1]>0))
    print('Num I:', np.sum(seir[2]>0))
    print('Num R', np.sum(seir[3]>0))

def print_seis_stats(seir):
    print('Num S:', np.sum(seir[0]>0))
    print('Num E', np.sum(seir[1]>0))
    print('Num I:', np.sum(seir[2]>0))
