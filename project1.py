#!/usr/bin/python3

"""
Michael Bronson
Alex Anthon
Calix Barrus



"""


import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

beta = -.0050367
p1c = .035
p1c = .7
p1c_fraction = p1c/(1-p1c)
exposed_mu = 1.57
exposed_sigma = 0.65
exposed_max_days = 15
infectious_mu = 2.25
infectious_sigma = 0.105
infectious_max_days = 14

def display_graph(g):
    nx.draw(g)
    plt.show()

def display_graph_from_numpy(matrix):
    g = nx.from_numpy_matrix(matrix)
    nx.draw(g)
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

def get_lognormal_max(mu, sigma, size, max_days_exposed):
    s = np.random.lognormal(mu, sigma, size)
    s[s > max_days_exposed] = max_days_exposed
    return s

def get_stats(seirs):
    seir = seirs[-1]
    return np.sum(seir[:, 0]>0), np.sum(seir[:, 1]>0), np.sum(seir[:, 2]>0), np.sum(seir[:, 3]>0)

def get_sus(seir):
    return np.sum(seir[:, 0]>0)

def print_stats(seir):
    print('Num S:', np.sum(seir[:, 0]>0))
    print('Num E', np.sum(seir[:, 1]>0))
    print('Num I:', np.sum(seir[:, 2]>0))
    print('Num R', np.sum(seir[:, 3]>0))


def simulate(matrix, seir, num_steps, masks=False, meeting_on_day=lambda x: True):
    """
    matrix: adjacency matrix of graph
    seir: starting numbers of s, e, i, r
    num_steps: number of steps to run for
    """
    seirs = np.zeros((num_steps, *seir.shape))
    days_exposed = get_lognormal_max(exposed_mu, exposed_sigma, len(matrix), exposed_max_days)
    days_infectious = get_lognormal_max(infectious_mu, infectious_sigma, len(matrix), infectious_max_days)
    for step in range(num_steps):
        # Probabilistic vales to use during the simulation
        probs = np.random.rand(len(matrix))
        # Infectious to Recoved
        to_r_filter = seir[:, 2] > days_infectious
        seir[to_r_filter, 3] = -1
        seir[to_r_filter, 2] = 0
        # Exposed to Infectious
        to_i_filter = seir[:, 1] > days_exposed
        seir[to_i_filter, 2] = -1
        seir[to_i_filter, 1] = 0
        # Susceptible to Exposed
        if meeting_on_day(step):
            i_filter = seir[:, 2] > 0
            # Getting infectious rates for each infectious person
            if masks:
                exponential = np.exp(beta * (seir[:, 2]**3 - 1))
                i_rates = 0.35 * (p1c_fraction*exponential)/(1+p1c_fraction*exponential)
            else:
                exponential = np.exp(beta * (seir[:, 2]**3 - 1))
                i_rates = (p1c_fraction*exponential)/(1+p1c_fraction*exponential)
            # import pdb; pdb.set_trace()
            to_e_probs = 1 - (np.prod((1 - (matrix * i_rates))[:, i_filter], axis=1))
            to_e_filter = (seir[:, 0] > 0) & (probs < to_e_probs)
            seir[to_e_filter, 1] = -1
            seir[to_e_filter, 0] = 0
        # Tracking days and seirs
        seir[(seir > 0)] += 1
        seir[(seir < 0)] = 1
        seirs[step] = seir

    # print_stats(seirs[-1])
    return seirs

# num_nodes = 40
# complete_graph = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
# complete_seir = get_seir(num_nodes, 1)
# num_steps = 500
# complete_seirs = simulate(complete_graph, complete_seir, num_steps)
# print(complete_seirs[-1])
# print_stats(complete_seirs[-1])

# cycle_graph = nx.to_numpy_matrix(nx.cycle_graph(num_nodes))
# cycle_graph = np.array(cycle_graph, dtype=np.float64)
# cycle_seir = get_seir(num_nodes, 1)
# cycle_seirs = simulate(cycle_graph, cycle_seir, num_steps)


def run_sims(matrix, masks=False, meeting_on_day=lambda x: True):
    seir = get_seir(len(matrix), 1)
    nums_sus = []
    for _ in range(1000):
        nums_sus.append(get_sus(simulate(matrix, np.array(seir), 100, masks, meeting_on_day)[-1]))
    nums_sus = sorted(nums_sus)
    max_sus = nums_sus[-1]
    min_sus = nums_sus[0]
    low_quarter_sus = nums_sus[int(.24*len(nums_sus))]
    median_sus = nums_sus[int(.49*len(nums_sus))]
    high_quarter_sus = nums_sus[int(.74*len(nums_sus))]
    mean_sus = sum(nums_sus) / len(nums_sus)
    
    return min_sus, low_quarter_sus, median_sus, high_quarter_sus, max_sus, mean_sus

cave_man_graph_40 = get_matrix_from_graph(nx.relaxed_caveman_graph(10, 4, 0.25))
display_graph_from_numpy(cave_man_graph_40)

cave_man_graph_100 = get_matrix_from_graph(nx.relaxed_caveman_graph(25, 4, 0.25))
display_graph_from_numpy(cave_man_graph_100)

print("Number of susceptible with following conditions.")
print("num_nodes, masks, days, (min, low_quarter, median, high_quarter, max, mean)")
print("40, no masks, every day:", run_sims(cave_man_graph_40))
print("10, no masks, every day:", run_sims(cave_man_graph_100))
print("40, with masks, every day:", run_sims(cave_man_graph_40, masks=True))
print("10, with masks, every day:", run_sims(cave_man_graph_100, masks=True))

weekdays = lambda x: (x % 7) < 5
print("40, no masks, weekdays:", run_sims(cave_man_graph_40, meeting_on_day=weekdays))
print("10, no masks, weekdays:", run_sims(cave_man_graph_100, meeting_on_day=weekdays))
print("40, with masks, weekdays:", run_sims(cave_man_graph_40, masks=True, meeting_on_day=weekdays))
print("10, with masks, weekdays:", run_sims(cave_man_graph_100, masks=True, meeting_on_day=weekdays))

mwf = lambda x: (x % 7) == 0 or (x % 7) == 2 or (x % 7) == 4
print("40, no masks, mwf:", run_sims(cave_man_graph_40, meeting_on_day=mwf))
print("10, no masks, mwf:", run_sims(cave_man_graph_100, meeting_on_day=mwf))
print("40, with masks, mwf:", run_sims(cave_man_graph_40, masks=True, meeting_on_day=mwf))
print("10, with masks, mwf:", run_sims(cave_man_graph_100, masks=True, meeting_on_day=mwf))

tt = lambda x: (x % 7) == 1 or (x % 7) == 2
print("40, no masks, tt:", run_sims(cave_man_graph_40, meeting_on_day=tt))
print("10, no masks, tt:", run_sims(cave_man_graph_100, meeting_on_day=tt))
print("40, with masks, tt:", run_sims(cave_man_graph_40, masks=True, meeting_on_day=tt))
print("10, with masks, tt:", run_sims(cave_man_graph_100, masks=True, meeting_on_day=tt))

