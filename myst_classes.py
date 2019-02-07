import networkx as nx
import re
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from itertools import count, product


class graph_solver:
    
    def __init__(self, edge_list, num_nodes, num_colors):
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.graph = self.build_graph()
        self.current_node = 0
        self.last_node = 0
        self.recursion_level = 0 # debug parameter
        self.data = self.find_solvable()
    
    def build_graph(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.num_nodes))
        nx.set_node_attributes(G, 0, name = 'color')
        G.add_edges_from(self.edge_list)
        return(G)
        
    def draw_current_config(self):
        colors = set(nx.get_node_attributes(self.graph, 'color').values())
        mapping = dict(zip(sorted(colors), count()))
        nodes = self.graph.nodes()
        colors = [mapping[self.graph.node[n]['color']] for n in nodes]

        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_edges(self.graph, pos)
        nc = nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, node_color=colors, 
                                    with_labels=True, cmap=plt.cm.jet)
        labels = nx.draw_networkx_labels(self.graph, pos)
        plt.colorbar(nc)
        plt.show()
        
    def iterate_color(self, node):
        current_color = nx.get_node_attributes(self.graph, 'color')[node]
        new_color = (current_color + 1) % self.num_colors
        nx.set_node_attributes(self.graph, {node : {'color' : new_color}})
        
    def reverse_iterate_color(self, node):
        current_color = nx.get_node_attributes(self.graph, 'color')[node]
        new_color = (current_color - 1) % self.num_colors
        nx.set_node_attributes(self.graph, {node : {'color' : new_color}})
        
    def set_all_colors(self, new_colors):
        new_color_dict = dict(enumerate([{'color' : color} for color in new_colors]))
        nx.set_node_attributes(self.graph, new_color_dict)
        
    def generate_all_configs(self):
        return list(product(range(0, self.num_colors), repeat = self.num_nodes))
    
    def backtrack(self, start_node, end_node):
        """Simulate a step in the puzzle, but backwards,
        reverse-iterating the color of the end node.

        Args:
            start_node (int): the node the step began from.
            end_node (int): the node the step ended on.

        Returns:
            bool : True if the backtrack was successful, False if the step can't be made
        """
        #an edge must exist to make the step
        if (start_node, end_node) not in self.graph.edges():
            return False
        #can't return to a node you just came from
        if(start_node == self.last_node):
            return False
        self.current_node = start_node
        self.last_node = end_node
        self.reverse_iterate_color(end_node)
        return True
    
    def search(self, configs, base_config):
        """Recursively find all solvable configurations for a graph by backtracking.
            Should only be called by its wrapper function, find_solvable

        Args:
            configs (set): set of solvable configurations found so far.
            base_config
        """
        current_config = tuple(nx.get_node_attributes(self.graph, 'color').values())
        print("All configs: " + str(configs))
        print("Current config is: " + str(current_config))
        if current_config not in configs:
            print("New config found!")
            configs.add(current_config)
            base_config = current_config
            base_current = self.current_node
            base_last = self.last_node
            for node in range(0, self.num_nodes):
                print("Trying to move from node " + str(self.current_node) + " to node " + str(node))
                if self.backtrack(node, self.current_node):
#                     print("Moving from node " + str(self.last_node) + " to node " + str(node))
                    print("Backtrack successful")
                    self.recursion_level += 1
                    print("Stepping down to level " + str(self.recursion_level))
                    print("Base config: " + str(base_config))
                    print("Base current: " + str(base_current))
                    print("Base lase: " + str(base_last))
                    # current_node is now node
                    # last_node is now the previous current_node
                    self.search(configs, current_config)
                    # when search bottoms out, reset node parameters
                    self.recursion_level -= 1
                    print("Stepping up to level " + str(self.recursion_level))
                    self.current_node = base_current
                    self.last_node = base_last
                    self.set_all_colors(base_config)
                else:
                    print("Could not backtrack")
        else:
            print("Repeat config found!")
                    
    def find_solvable(self):
        print("Solving graph with:")
        print("Nodes: " + str(self.num_nodes))
        print("Colors: " + str(self.num_colors))
        all_configs = self.generate_all_configs()
        data = pd.DataFrame(columns = ['Node', 'Solvable_Flag', 'Configs', 'Num_Configs'])
        all_configs_set = set(all_configs)
        threshold = (self.num_colors ** self.num_nodes) / 2
        print("This graph has " + str(len(all_configs)) + " possible configs; the threshold is " + str(threshold))
        # attempt puzzle from each possible end point
        for node in range(0, self.num_nodes):
            d = {}
            print()
            print("Finding solutions starting from node " + str(node) + ":")
            # reset graph to solved config
            self.set_all_colors(all_configs[0])
            self.recursion_level = 0
            # create fresh solvable list
            solvable = set()
            self.current_node = node
            self.last_node = node
            self.search(solvable, all_configs[0])
            print("Search Completed")
            d['Node'] = [node]
            # determine whether more than half of all configs are solvable 
            # if more than half are solvable, save only unsolvable configs and set solvable flag to 0
            if len(solvable) > threshold:
                unsolvable = all_configs_set - solvable
                num_configs = len(unsolvable)
                d['Solvable_Flag'] = [0]
                d['Configs'] = [unsolvable]
                d['Num_Configs'] = num_configs
            # otherwise, save all solvable configs and set solvable flag to 1
            else:
                num_configs = len(solvable)
                d['Solvable_Flag'] = [1]
                d['Configs'] = [solvable]
                d['Num_Configs'] = num_configs
            row = pd.DataFrame(data = d)
            data = data.append(row, ignore_index = True)
        return data

class configuration:

    def __init__(self, direction, config):
        self.direction = direction
        self.config = config

    def __eq__(self, other):
        if isinstance(other, configuration):
            return (self.direction == other.direction) and (self.config == other.config)
       
    def __str__(self):
        return "Configuration " + str(self.config) + "from node " + str(self.direction)