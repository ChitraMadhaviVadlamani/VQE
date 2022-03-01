# useful additional packages
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import numpy as np
import networkx as nx

from qiskit import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram
#################################################2
def draw_graph(G, colors, pos): #draws undirected graph
    default_axes = plt.axes(frameon=True) #Whether the Axes frame is visible.
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos) #draws graph
    print(nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos))
    edge_labels = nx.get_edge_attributes(G, "weight")
    print(edge_labels,"$$$")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels) #adds edges
    #Edge labels in a dictionary keyed by edge two-tuple of text labels (default=None). Only labels for the keys in the dictionary are drawn.
###########################1
n = 3
tsp = Tsp.create_random_instance(n, seed=124) #Seed helps determine weight of edges between nodes
print(tsp.graph,"HERE1")
adj_matrix = nx.to_numpy_matrix(tsp.graph)
print("distance\n", adj_matrix)

colors = ["r" for node in tsp.graph.nodes] #color of nodes
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes] #A dictionary with nodes as keys and positions as values. Positions should be sequences of length 2.
print(pos,"@@ ",colors)
draw_graph(tsp.graph, colors, pos)
#####################################4


from itertools import permutations


def brute_force_tsp(w, N): #w is adj matrix and N is no of nodes
    a = list(permutations(range(1, N)))  #(1,2) adn (2,1)
    last_best_distance = 1e10
    for i in a:
        print(i," i value")
        distance = 0
        pre_j = 0 #ex. i =(1,0)--> j=1 then j = 0
        for j in i:
            print(j, " j value")
            distance = distance + w[j, pre_j]
            print(distance," ABCD")
            pre_j = j
        distance = distance + w[pre_j, 0]
        print(distance," EFGH")
        order = (0,) + i #order of nodes travlled
        print(order," order")
        if distance < last_best_distance:
            best_order = order
            last_best_distance = distance
            print("order = " + str(order) + " Distance = " + str(distance))
    return last_best_distance, best_order
############3

best_distance, best_order = brute_force_tsp(adj_matrix, n)
print(
    "Best order from brute force = "
    + str(best_order)
    + " with total distance = "
    + str(best_distance)
)
###################5

def draw_tsp_solution(G, order, colors, pos):  #draws graph with arrows!
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)


draw_tsp_solution(tsp.graph, best_order, colors, pos)

#########################
