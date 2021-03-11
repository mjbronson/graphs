#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seir_sim
import sei_sim
import sim_helper as sh
import visual
import small_world_caveman
import get_graphs as gg

def simulate_graphs(matrices, num_sims):
    info = []
    for m in matrices:
        info.append(simulate_graph(m, num_sims))
    return info

def simulate_graph(m, num_sims):
    total_good_exposure = 0
    total_bad_exposure = 0
    for _ in range(num_sims):
        seirs = seir_sim.quick_sim(m)
        seis = sei_sim.quick_sim(m)[:len(seirs)]
        good_exposure = sh.seis_exposed(seis)
        bad_exposure = sh.seirs_exposed(seirs)
        total_good_exposure += good_exposure
        total_bad_exposure += bad_exposure
    return total_good_exposure/num_sims, total_bad_exposure/num_sims


def simulate_graph_and_display(m, layout=None):
    seirs = seir_sim.quick_sim(m)
    seis = sei_sim.quick_sim(m)[:len(seirs)]
    visual.visualize(m, sim_info1=seirs, n1=visual.seir_nodes, sim_info2=seis, e2=visual.sei_edges, layout=layout, time_delay=0.001)
    good_exposure = sh.seis_exposed(seis)
    bad_exposure = sh.seirs_exposed(seirs)
    return good_exposure, bad_exposure


def sim_alex_graphs():
    num_sims = 500
    matrices = [
        gg.read_adj_list('../irn-tui/graphs/cavemen-50-10.txt'),
        gg.read_adj_list('../irn-tui/graphs/complete-500.txt'),
        gg.read_adj_list('../irn-tui/graphs/hex-lattice-15x15.txt'),
        gg.read_adj_list('../irn-tui/graphs/line-graph.txt'),
        gg.read_adj_list('../irn-tui/graphs/spatial-network.txt'),
        gg.read_adj_list('../irn-tui/graphs/square-lattice.txt'),
        gg.read_adj_list('../irn-tui/graphs/triangle-lattice.txt')
    ]
    titles = [
        'cavemen',
        'complete',
        'hex lattice',
        'line graph',
        'spatial network',
        'square lattice',
        'triangle lattice'
    ]
    info = simulate_graphs(matrices, num_sims)

    print('{t:20} {n:>7} {x:>7} {y:>7}'.format(t='graph type', n='ratio', x='good', y='bad'))
    for t, i in zip(titles, info):
        print('{t:20} {n:7.2f} {x:7} {y:7}'.format(t=t, n=i[0]/i[1], x=i[0], y=i[1]))



if __name__ == '__main__':
    matrices = [
        gg.read_adj_list_with_layout('../irn-tui/graphs/cavemen-50-10.txt'),
        gg.read_adj_list_with_layout('../irn-tui/graphs/complete-500.txt'),
        gg.read_adj_list_with_layout('../irn-tui/graphs/hex-lattice-15x15.txt'),
        gg.read_adj_list_with_layout('../irn-tui/graphs/line-graph.txt'),
        gg.read_adj_list_with_layout('../irn-tui/graphs/spatial-network.txt'),
        gg.read_adj_list_with_layout('../irn-tui/graphs/square-lattice.txt'),
        gg.read_adj_list_with_layout('../irn-tui/graphs/triangle-lattice.txt')
    ]

    # print(len(m))
    # sh.display_graph_from_numpy(m, l)
    # simulate_graph_and_display(m, l)
    # simulate_graphs(matrices, 500)
    # for m, l in matrices[:]:
    #     sh.display_graph_from_numpy(m, l)
    m = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ])
    g = nx.from_numpy_array(m)
    nx.draw(g)
    plt.show()