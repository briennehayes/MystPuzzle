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
        data (pd.DataFrame): describes which configurations are solvable ending on each node in the graph.
        num_solvable (int): total number of configurations that are solvable in this graph, from any node.

    """
    # TODO: autosolve should be a parameter
    def __init__(self, edge_list, num_nodes, num_colors):
        self.edge_list = edge_list
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.graph = self.build_graph()
        self.current_node = 0
        self.last_node = 0
        # These values are computed up-front since they're used in find_solvable
        self.all_color_sets = self.generate_all_color_sets()
        self.all_configs = self.generate_all_configs()
        self.all_configs_dict = self.generate_all_configs_dict()
        # Puzzle solutions are calculated on creation
        self.data = self.find_solvable()
        self.num_solvable = self.data['Num_Color_Sets']
        self.expanded = self.build_expanded_graph()
    
    def build_graph(self):
        """Generates a networkx graph object w/ color attributes for this puzzle instance.
        """
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.num_nodes))
        G.add_node(-1) # starting position, not an actual node
        nx.set_node_attributes(G, 0, name = 'color')
        start_edges = [(-1, num) for num in np.arange(self.num_nodes)]
        G.add_edges_from(self.edge_list)
        G.add_edges_from(start_edges)
        return(G)
        
    def draw_current_config(self):
        """Visualize the puzzle's graph and current color configuration.
        """
        G = self.graph.subgraph(np.arange(self.num_nodes))

        colors = set(nx.get_node_attributes(G, 'color').values())
        mapping = dict(zip(sorted(colors), count()))
        nodes = G.nodes()
        colors = [mapping[G.node[n]['color']] for n in nodes]

        pos = nx.spring_layout(G)
        nx.draw_networkx_edges(G, pos)
        nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
                                    with_labels=True, cmap=plt.cm.jet)
        labels = nx.draw_networkx_labels(G, pos)
        plt.colorbar(nc) # TODO: make colorbar discrete / show all color values
        plt.show()

    def draw_expanded_graph(self):
        """Visualize the puzzle's graph and current color configuration.
        """
        G = self.expanded
        nodes = G.nodes()

        # pos = nx.spring_layout(G)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_edges(G, pos)
        nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, with_labels=True)
        labels = nx.draw_networkx_labels(G, pos)
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
        new_color_dict = dict(enumerate([{'color' : color} for color in new_colors]))
        new_color_dict[-1] = {'color' : 0} # color of start position doesn't actually matter
        nx.set_node_attributes(self.graph, new_color_dict)

    def set_current_config(self, new_config):
        """Set the current configuration of the solver.

        Args:
            new_config (configuration): the desired configuration.
        """
        self.set_all_colors(new_config.color_set)
        self.current_node = new_config.current
        self.last_node = new_config.last

    def get_current_config(self):
        """Returns the current configuration of the solver.

        Returns:
            configuration: the solver's current configuration
        """
        G = self.graph.subgraph(np.arange(self.num_nodes))

        return configuration(self.current_node, 
                             self. last_node,
                             tuple(nx.get_node_attributes(G, 'color').values()))

        
    def generate_all_color_sets(self):
        """Produce a list of all possible color sets (not just solvable) for this puzzle instance.

        Returns:
            list of tuples of ints: all possible color sets for this puzzle instance.
        """
        return list(product(range(0, self.num_colors), repeat = self.num_nodes))

    # TODO rewrite this function to make it more efficient (list comprehensions?)
    def generate_all_configs(self):
        """Produce a list of all possible configurations (not just solvable) for this puzzle instance.

        Returns:
            list of configurations: all possible configurations for this puzzle instance.
        """
        # specifically refer to solved color set 
        zeros = self.all_color_sets[0]

        # determine all possible moves
        moves = []
        for node in range(0, self.num_nodes):
            moves.append((-1, node)) # every node has the -1 connection

        for curr_node in range(0, self.num_nodes):
            for prev_node in range(0, self.num_nodes):
                if (curr_node, prev_node) in self.graph.edges():
                    moves.append((curr_node, prev_node))
        
        # now combine them
        all_configs = []
        all_configs.append(configuration(-1, -1, zeros))
        for move in moves:
            if -1 in move:
                all_configs.append(configuration(move[0], move[1], zeros))
            else:
                for color_set in self.all_color_sets:
                    all_configs.append(configuration(move[0], move[1], color_set))

        return all_configs

    def generate_all_configs_dict(self):
        # build a dictionary of all configurations and integer indices
        # need to build the relation both ways
        ints = range(0, len(self.all_configs))
        num_to_config = dict(zip(ints, self.all_configs))
        config_to_num = dict(zip(self.all_configs, ints))
        all_configs_dict = {**num_to_config, **config_to_num}
        return all_configs_dict

    
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
    
    def search(self, final_config, verbose = False): 
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

        if verbose:
            print("Starting with configuration " + str(final_config))
  
        while queue: 
  
            # Dequeue next config and set the puzzle
            base_config = queue.pop(0)
            self.set_current_config(base_config)

            if verbose:
                print("The base config is " + str(base_config))
  
            # Attempt to backtrack to every adjacent configuration, 
            # saving and enqueueing new configs as they're discovered
            for node in range(0, self.num_nodes):
                if verbose:
                    print("Attempting backtrack to node " + str(node))
                if self.backtrack(node, self.current_node):
                    if verbose:
                        print("Backtrack successful!")
                    current_config = self.get_current_config()
                    if verbose:
                        print("New config is " + str(current_config))
                    if current_config not in solvable:
                        queue.append(current_config)
                        solvable.add(current_config)
                    self.set_current_config(base_config)
                    if verbose:
                        print("Resetting to base config")
                else:
                    if verbose:
                        print("Cannot backtrack")
        
        return solvable

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

        # these are referred to later 
        all_color_sets_set = set(self.all_color_sets)
        threshold = (self.num_colors ** self.num_nodes) / 2 # this will be used to determine which configs get saved

        # this will store the graph's solvability information
        data = pd.DataFrame(columns = ['Solvable_Flag', 'Color_Sets', 'Num_Color_Sets'])
        
        if verbose:
            print("This graph has " + str(len(self.all_color_sets)) + " possible configs; the threshold is " + str(threshold))
        
        initial_config = configuration(-1, -1, self.all_color_sets[0])

        solvable = self.search(initial_config)

        solvable_color_sets = set()
        for config in solvable:
            solvable_color_sets.add(config.color_set)

        if verbose:
            print("Search Completed")

        d = {}
        # determine whether more than half of all configs are solvable 
        # if more than half are solvable, save only unsolvable configs and set solvable flag to 0
        # since most graphs are entirely solvable, this saves a lot of space
        if len(solvable_color_sets) > threshold:
            unsolvable = all_color_sets_set - solvable_color_sets
            num_color_sets = len(unsolvable)
            d['Solvable_Flag'] = [0]
            d['Color_Sets'] = [unsolvable]
            d['Num_Color_Sets'] = num_color_sets
        # otherwise, save all solvable configs and set solvable flag to 1
        else:
            num_color_sets = len(solvable_color_sets)
            d['Solvable_Flag'] = [1]
            d['Color_Sets'] = [solvable]
            d['Num_Color_Sets'] = num_color_sets

        if verbose:
            print("This puzzle has " + str(num_color_sets) + " solvable color sets.")
        
        data = pd.DataFrame(data = d)
        return data

    # TODO: this method is awful, rewrite it
    def is_reachable(self, start, end):
        # special case: moving from solved config to exit config
        if start.is_solved():
            if start.current != -1:
                return end.current == -1 and end.last == start.current
            else:
                return end.current == -1 and end.last == -1
        # first, check to make sure the move from start to end makes sense
        if (start.current == end.last) and (start.last != end.current) and (start.current, end.current) in self.graph.edges():
            # now we need to ensure the change in color sets is possible
            diffs = []
            for i in range(0, len(start.color_set)):
                diffs.append(start.color_set[i] - end.color_set[i])
            not_zeros = [diff != 0 for diff in diffs]
            # color sets should only differ by one place
            if sum(not_zeros) == 1:
                not_zero_index = [index for index, val in enumerate(not_zeros) if val][0]
                # where they differ, end's color should equal start's color +1 mod num_colors
                return end.color_set[not_zero_index] == ((start.color_set[not_zero_index] + 1) % self.num_colors)
            else:
                return False
        else:
            return False

    # TODO documentation
    def build_expanded_graph(self):
        expanded = nx.DiGraph()
        for config in self.all_configs:
            expanded.add_node(self.all_configs_dict[config], config = config)
        for start_config in self.all_configs:
            for end_config in self.all_configs:
                if self.is_reachable(start_config, end_config):
                    expanded.add_edge(self.all_configs_dict[start_config], self.all_configs_dict[end_config])

        return expanded



class configuration:
    """Stores information about a single graph configuration, including directionality and color.

    Attributes:
        current (int): the location of the solver in this configuration.
        last (int): where the solver came from in this configuration.
        color_set (tuple of ints): ordered representation of the color states of each node in the current configuration.
    """

    def __init__(self, current, last, color_set):
        self.current = current
        self.last = last
        self.color_set = color_set

    def is_solved(self):
        return self.color_set == tuple([0] * len(self.color_set))

    def __eq__(self, other):
        # TODO: else statement to throw an error for incompatible types?
        if isinstance(other, configuration):
            return ((self.current == other.current) and
                    (self.last == other.last) and
                    (self.color_set == other.color_set))
       
    def __str__(self):
        return "Configuration " + str(self.color_set) + " at node " + str(self.current) + " from node " + str(self.last)

    # # TODO: write a more efficient hash function (that doesn't break)
    # # USE tuple hash function
    # def __hash__(self):
    #     # obviously this breaks for graphs with 100 nodes, so this is a temporary fix
    #     curr = 99 if self.current == -1 else self.current
    #     prev = 99 if self.last == -1 else self.last

    #     hash = str(curr) + str(prev)
    #     for num in self.color_set:
    #         hash += str(num)
    #     return int(hash)

    def __hash__(self):
        tup = (self.current, self.last, self.color_set)
        return hash(tup)