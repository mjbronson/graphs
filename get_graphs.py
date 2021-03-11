
from networkx.drawing import layout
import numpy as np
import networkx as nx
import seir_sim
import sei_sim
import sim_helper as sh
import visual
import small_world_caveman

def get_complete():
    return lambda x: nx.complete_graph(x)


def read_adj_list(file_name) -> np.ndarray:
    """
    This reads in the data from half a symmetric matrix and mirrors it.
    If the whole matrix is present in the file, that won't cause problems.
    This cannot read unsymmetric matrices.
    """
    with open(file_name, 'r') as f:
        line = f.readline()
        shape = (int(line[:-1]), int(line[:-1]))
        matrix = np.zeros(shape, dtype=np.uint8)

        line = f.readline()[:-1]
        while len(line) > 0:
            coord = line.split(' ')
            matrix[int(coord[0]), int(coord[1])] = 1
            matrix[int(coord[1]), int(coord[0])] = 1
            line = f.readline()[:-1]
    return matrix


def read_adj_list_with_layout(file_name) -> np.ndarray:
    """
    This reads in the data from half a symmetric matrix and mirrors it.
    If the whole matrix is present in the file, that won't cause problems.
    This cannot read unsymmetric matrices.
    """
    with open(file_name, 'r') as f:
        line = f.readline()
        shape = (int(line[:-1]), int(line[:-1]))
        matrix = np.zeros(shape, dtype=np.uint8)

        line = f.readline()[:-1]
        while len(line) > 0:
            coord = line.split(' ')
            matrix[int(coord[0]), int(coord[1])] = 1
            matrix[int(coord[1]), int(coord[0])] = 1
            line = f.readline()[:-1]
        layout = {}
        line = f.readline()[:-1]
        while len(line) > 0:
            fields = line.split(' ')
            node = int(fields[0])
            x = float(fields[1])
            y = float(fields[2])
            layout[node] = (x, y)
            line = f.readline()[:-1]
    return matrix, layout