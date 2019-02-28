import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count, product


class graph_solver:
    """Finds all solvable configurations of a graph puzzle instance.

    Attributes:
        edge_list (list of tuples of ints): description of the graph's edges.
        num_nodes (int): how many nodes/vertices the graph has.
        num_colors (int): how many distinct colors are to be used for this puzzle instance.
        graph (networkx graph): graph object (w/ color attributes) for this puzzle instance (private?).
        current_node (int): current location of the solver (private?).
        last_node (int): previous location of the solver (private?).
        recursion_level (int): tracks how deeply the solver recurs (for debugging; will be removed when non-recursive search is implemented)
        data (pd.DataFrame): describes which configurations are solvable ending on each node in the graph.
        num_solvable (int): total number of configurations that are solvable in this graph, from any node.

    """
    
    def __init__(self, edge_list, num_nodes, num_colors):
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.graph = self.build_graph()
        self.current_node = 0
        self.last_node = 0
        self.recursion_level = 0 # debug parameter
        self.data = self.find_solvable()
        self.num_solvable = self.get_num_solvable()
    
    def build_graph(self):
        """Generates a networkx graph object w/ color attributes for this puzzle instance.
        """
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.num_nodes))
        nx.set_node_attributes(G, 0, name = 'color')
        G.add_edges_from(self.edge_list)
        return(G)
        
    def draw_current_config(self):
        """Visualize the puzzle's graph and current color configuration.
        """
        colors = set(nx.get_node_attributes(self.graph, 'color').values())
        mapping = dict(zip(sorted(colors), count()))
        nodes = self.graph.nodes()
        colors = [mapping[self.graph.node[n]['color']] for n in nodes]

        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_edges(self.graph, pos)
        nc = nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes, node_color=colors, 
                                    with_labels=True, cmap=plt.cm.jet)
        labels = nx.draw_networkx_labels(self.graph, pos)
        plt.colorbar(nc) # TODO: make colorbar discrete / show all color values
        plt.show()
        
    # TODO: make internal solving/backtracking methods private?

    def iterate_color(self, node):
        """Advance the color of a single node by 1 (for use in solving).

        Args:
            node (int): which node is to be changed.
        """
        current_color = nx.get_node_attributes(self.graph, 'color')[node]
        new_color = (current_color + 1) % self.num_colors
        nx.set_node_attributes(self.graph, {node : {'color' : new_color}})
        
    def reverse_iterate_color(self, node):
        """Decrement the color of a single node by 1 (for use in backtracking).

        Args:
            node (int): which node is to be changed.
        """
        current_color = nx.get_node_attributes(self.graph, 'color')[node]
        new_color = (current_color - 1) % self.num_colors
        nx.set_node_attributes(self.graph, {node : {'color' : new_color}})
        
    def set_all_colors(self, new_colors):
        """Reset the colors of all nodes to a provided configuration.

        Args:
            new_colors (tuple of ints): the graph's new configuration.
        """
        # TODO: accept configuration objects and strip out the direction parameters
        new_color_dict = dict(enumerate([{'color' : color} for color in new_colors]))
        nx.set_node_attributes(self.graph, new_color_dict)

    def set_current_config(self, new_config):
        """Set the current configuration of the solver.

        Args:
            new_config (configuration): the desired configuration.
        """
        self.set_all_colors(base_config.config)
        self.current_node = new_config.current
        self.last_node = new_config.last

    def get_current_config(self):
        """Returns the current configuration of the solver.

        Returns:
            configuration: the solver's current configuration
        """
        return configuration(self.current_node, 
                             self. last_node,
                             tuple(nx.get_node_attributes(self.graph, 'color').values()))

        
    def generate_all_configs(self):
        """Produce a list of all possible configurations (not just solvable) for this puzzle instance.

        Returns:
            list of tuples of ints: all possible configurations for this puzzle instance.
        """
        return list(product(range(0, self.num_colors), repeat = self.num_nodes))
    
    def backtrack(self, start_node, end_node):
        """Simulate a step in the puzzle, but backwards, reverse-iterating the color of the end node.

        Args:
            start_node (int): the node the step began from.
            end_node (int): the node the step ended on.

        Returns:
            bool : True if the backtrack was successful, False if the step can't be made
        """
        # TODO: should just use current_node as start_node

        # an edge must exist to make the step
        if (start_node, end_node) not in self.graph.edges():
            return False
        # can't return to a node you just came from
        if(start_node == self.last_node):
            return False
        self.current_node = start_node
        self.last_node = end_node
        self.reverse_iterate_color(end_node)
        return True
    
    def search(self, final_config): 
        """Find all solvable configurations for a graph via backtracking using a breadth-first search.
            Should only be called by its wrapper function, find_solvable(). 

        Args:
            final_config (configuration): the ending configuration of the puzzle, from which backtracking starts.

        Returns:
            set of configurations: all solvable configurations for this graph.
        """
  
        # For storing discovered solvable configs 
        solvable = set()
  
        queue = [] 
  
        # Begin by adding the final (starting) config
        queue.append(final_config) 
        solvable.add(final_config)
  
        while queue: 
  
            # Dequeue next config and set the puzzle
            base_config = queue.pop(0)
            self.set_current_config(base_config)
  
            # Attempt to backtrack to every adjacent configuration, s
            # saving and enqueueing new configs as they're discovered
            for node in range(0, self.num_nodes):
                if self.backtrack(node, self.current_node):
                    current_config = self.get_current_config()
                    if current_config not in solvable:
                        queue.append(current_config)
                        solvable.add(current_config)
                    self.set_current_config(base_config)
        
        return solvable

    # This is the find_solvable for the breadth-first search method
    def find_solvable(self, verbose = False):
        """Executes a search ending on every node in the graph, then stores the results.

        Args:
            verbose (bool): if True, prints out the steps of the solving algorithm.

        Returns:
            pd.DataFrame: which configurations are solvable ending on each node in the graph.
        """
        if verbose:
            print("Solving graph with:")
            print("Nodes: " + str(self.num_nodes))
            print("Colors: " + str(self.num_colors))
        all_configs = self.generate_all_configs()
        data = pd.DataFrame(columns = ['Node', 'Solvable_Flag', 'Configs', 'Num_Configs'])
        all_configs_set = set(all_configs)
        threshold = (self.num_colors ** self.num_nodes) / 2 # this will be used to determine which configs get saved
        if verbose:
            print("This graph has " + str(len(all_configs)) + " possible configs; the threshold is " + str(threshold))
        # attempt puzzle from each possible end point
        for node in range(0, self.num_nodes):
            d = {}
            if verbose:
                print()
                print("Finding solutions starting from node " + str(node) + ":")
            # reset graph to solved config
            starting_config = configuration(node, node, all_configs[0])
            self.set_current_config(starting_config)
            config_set = self.search(starting_config)
            solvable = set()
            # strip directions off of configurations
            for config in config_set:
                solvable.add(config.config)
            if verbose:
                print("Search Completed")
            d['Node'] = [node]
            # determine whether more than half of all configs are solvable 
            # if more than half are solvable, save only unsolvable configs and set solvable flag to 0
            # since most graphs are entirely solvable, this saves a lot of space
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

    # TODO would it be more efficient to calculate this number in find_solvable?
    def get_num_solvable(self):
        """Determine the total number of solvable configurations for this puzzle instance.

        Returns:
            int: how many configurations are solvable in this puzzle instance.
        """
        all_configs = set(self.generate_all_configs())

        solvable = set()

        for index in self.data.index:
            points = self.data.loc[index, ['Solvable_Flag', 'Configs']]
            if points['Solvable_Flag'] == 0:
                solvable = solvable.union(all_configs - points['Configs'])
            else:
                solvable = solvable.union(points['Configs'])
                
        return len(solvable)


class configuration:
    """Stores information about a single graph configuration, including directionality and color.

    Attributes:
        current (int): the location of the solver in this configuration.
        last (int): where the solver came from in this configuration.

    """

    def __init__(self, current, last, config):
        self.current = current
        self.last = last
        self.config = config

    def __eq__(self, other):
        if isinstance(other, configuration):
            return ((self.current == other.current) and
                    (self.last == other.last) and
                    (self.config == other.config))
       
    def __str__(self):
        return "Configuration " + str(self.config) + " at node " + str(self.current) + " from node " + str(self.last)

    # TODO: write a more efficient hash function
    def __hash__(self):
        hash = str(self.current) + str(self.last)
        for num in self.config:
            hash += str(num)
        return int(hash)