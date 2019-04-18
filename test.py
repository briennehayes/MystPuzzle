import myst_classes as mc
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# The three-vertex graph

graph3 = mc.graph_solver([(0, 1), (1, 2), (2, 0)], 3, 2)

graph3.draw_current_config()
graph3.draw_expanded_graph()


# Four-vertex graphs

four_graphs = pd.read_csv('graph_lists/four_graphs.csv', index_col = 0)
four_graphs['Edges'] = [eval(edge_list) for edge_list in four_graphs['Edges']]

graph4_4 = mc.graph_solver(four_graphs.loc[3, 'Edges'], 4, 2)
graph4_5 = mc.graph_solver(four_graphs.loc[4, 'Edges'], 4, 2)
graph4_6 = mc.graph_solver(four_graphs.loc[5, 'Edges'], 4, 2)

graph4_4.draw_current_config()
graph4_4.draw_expanded_graph()

graph4_5.draw_current_config()
graph4_5.draw_expanded_graph()

graph4_6.draw_current_config()
graph4_6.draw_expanded_graph()