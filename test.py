import myst_classes as mc
import pandas as pd

four_graphs = pd.read_csv('graph_lists/four_graphs.csv', index_col = 0)
four_graphs['Edges'] = [eval(edge_list) for edge_list in four_graphs['Edges']]

graph4_4 = mc.graph_solver(four_graphs.loc[3, 'Edges'], 4, 2)
graph4_5 = mc.graph_solver(four_graphs.loc[4, 'Edges'], 4, 2)
graph4_6 = mc.graph_solver(four_graphs.loc[5, 'Edges'], 4, 2)