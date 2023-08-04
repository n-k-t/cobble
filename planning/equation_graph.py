from operators import *

import networkx as nx
import os

operator_labels = {
    Addition: 'ADD', 
    Subtraction: 'SUB', 
    MatrixMultiplication: 'MATMUL'
}

# Will break if there are less than two parents. Should fix for when activation functions are added in.
def to_svg(root) -> 'None':
    topological = root.topological_sort()
    topological.reverse()
    G = nx.DiGraph()
    for count, tensor in enumerate(topological):
        parents = [*tensor.operator.parents]
        G.add_edge(str(parents[0]), str(tensor), label = operator_labels[tensor.operator.__class__])
        G.add_edge(str(parents[1]), str(tensor), label = operator_labels[tensor.operator.__class__])
        if count == 0:
            G.nodes[str(tensor)]['fillcolor'] = '#FFF000'
            G.nodes[str(tensor)]['color'] = '#000000'
            G.nodes[str(tensor)]['style'] = 'filled, dashed'
            G.nodes[str(tensor)]['label'] = 'Output'
        else:
            G.nodes[str(tensor)]['fillcolor'] = '#FFCCCB'
            G.nodes[str(tensor)]['color'] = '#000000'
            G.nodes[str(tensor)]['style'] = 'filled'
            G.nodes[str(tensor)]['label'] = 'Tensor'
        for i in parents:
            if i.operator == None:
                G.nodes[str(i)]['fillcolor'] = '#32CD32'
                G.nodes[str(i)]['color'] = '#000000'
                G.nodes[str(i)]['style'] = 'filled'
                G.nodes[str(i)]['label'] = 'Instantiate'
    nx.drawing.nx_pydot.write_dot(G, f'equation_graph.dot')
    os.system(f'dot -Tsvg equation_graph.dot -o equation_graph.svg')