# tools for visualizeing graphs


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from . import utils
class GraphViz:
    nodesize = 4800
    nodemarker = "o"
    axis_extension_factor = 1.1
    def __init__(self, network_table, node_names, width_range=(2,20)):
        """
        Args:
            network_table (np.array, symmetric): adjacency matrix of the network
            node_names (_type_): the names of the nodes corresponding to the table
        """
        dim1, dim2 = network_table.shape
        if ( dim1 != dim2 ) or not np.allclose(network_table, network_table.T, atol=0.1, rtol=0.1):
            raise ValueError('network_table should be (roughly) symmetric matrix')
        if dim1 != len(node_names):
            raise ValueError('the number of node_names is not the same as the size of the network_table')

        self.network_table = self.process_network_table(network_table)
        self.node_names = node_names

        self.min_width, self.max_width = width_range
        
        self.make_graph()
        
    def process_network_table(self, network_table):
        symmetrized_table = (network_table + network_table.T)/2

        lower_graph_table = np.tril(symmetrized_table, k=-1)
        return lower_graph_table
        
    def sorted_weights(self):
        table = self.network_table
        sorted_values = -np.sort(-table[table!=0])
        return sorted_values

    
    def knee_plot(self, plot_threshold=None):
        plt.rcParams["font.family"] = utils.get_config()['plot_params']['font']
        plt.rcParams['font.size'] = utils.get_config()['plot_params']['font_size']
        width = utils.get_config()['plot_params']['paper_width']

        fig, ax = plt.subplots(dpi=300, figsize=(width, 0.6*width))

        weights = self.sorted_weights()
        idx = np.arange(1, len(weights)+1)
        ax.plot(idx, weights, 'x')
        ax.set_xticks(idx[::2])
        ax.set_xlabel('Edge Number')
        plt.rcParams['text.usetex'] = True
        ax.set_ylabel(r'$\parallel d(\alpha) \parallel$')
        plt.rcParams['text.usetex'] = False
        if plot_threshold is not None:
            float_threshold = self._get_float_threshold(plot_threshold)
            ax.axhline(y = float_threshold, color = 'r', linestyle = ':')
        ax.set_ylim([0, 0.5])
        return ax

    
        
    def make_graph(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(self.node_names)
        iinds, jinds = np.nonzero(self.network_table > 0)
        
        for i, j in zip(iinds, jinds):
            iName = self.node_names[i]
            jName = self.node_names[j]
            self.G.add_edge(iName, jName, weight=self.network_table[i,j])
            
        return self.G

    def _get_float_threshold(self, threshold):
        """
        Args:
            threshold (int or float): int: number of edges to show, float, the threshold
        Returns:
            float
        """
        if type(threshold) is int:
            # plot the largest number of edges
            edge_and_weights = nx.get_edge_attributes(self.G, 'weight')
            plot_threshold = (self.sorted_weights()[threshold-1] + self.sorted_weights()[threshold])/2
        else:
            plot_threshold = threshold
            
        return plot_threshold

    def top_edges(self, threshold, return_values=False):
        float_threshold = self._get_float_threshold(threshold)

        edge_and_weights = nx.get_edge_attributes(self.G, 'weight')

        collector = list()
        weights = list()

        for edge, weight in edge_and_weights.items():
            if weight >= float_threshold:
                edge_s = sorted(edge)

                formatted = "{}-{}".format(edge_s[0], edge_s[1])
                collector.append(formatted)
                weights.append(weight)

        if return_values:
            return (collector, weights)
        else:
            return set(collector)



    def _weight_display_widths(self, weights):
        """
        Args:
            weights (list[float]): list of the weights for the edges to be displayed

        Returns:
            thicknesses (list[float]): a same length list with the weight of each edge to display
        """
        weightss = np.array(weights)
        weights = None

        minw = np.min(weightss)
        maxw = np.max(weightss)
        standardised_w =  (weightss - minw)/(maxw-minw)

        thicknesses = standardised_w * (self.max_width - self.min_width) + self.min_width
        return thicknesses.tolist()


    def show_graph(self, plot_threshold, small_figure=False, use_params_from_config=True):
        """ make a plot visualizing the graph

        Args:
            plot_threshold (int or float): if int, the number of edges to show, if float, the threshold above which to show an edge
            small_figure (boolean): whether the figure should be small, then font-size is doubled and synonyms are used
            use_params_from_config: whether to read category colors and names from the config.json file
        """
        
    
        font_size = utils.get_config()['plot_params']['font_size']

        if small_figure:
            font_size = font_size*2.5

        width = utils.get_config()['plot_params']['paper_width']
        if use_params_from_config:
            dpi=utils.get_config()['plot_params']['dpi']
        else:
            dpi=None
        
        fig, ax = plt.subplots(figsize=(width,0.9*width), dpi=dpi)
        ax.axis('off')
        edge_and_weights = nx.get_edge_attributes(self.G, 'weight')
        

        float_threshold = self._get_float_threshold(plot_threshold)
        
        edges = list()
        thickness = list()
        weights = list()
        for edge, weight in edge_and_weights.items():
            if weight >= float_threshold:
                edges.append(edge)
                
                weights.append(weight)
            
        thickness = self._weight_display_widths(weights)
        nodelist = self.G.nodes

        if small_figure:
            category_rename_dict = utils.get_config()['category_synonyms']
        else:
            category_rename_dict = utils.get_category_rename_dict(linebreaks=2)
        
        if use_params_from_config:
            rename_dict_this = {node: category_rename_dict[node] for node in nodelist}
            cat_colors = utils.get_config()['category_colors']
            node_colors = [cat_colors[n] for n in nodelist]
        else:
            rename_dict_this = None 
            node_colors = None
        
        nx.draw_networkx(self.G,
                    nodelist=nodelist,
                    edgelist=edges,
                    width = thickness,
                    pos = nx.circular_layout(nodelist),
                    font_size = font_size,
                    labels = rename_dict_this,
                    node_size = self.nodesize,
                    node_shape = self.nodemarker,
                    node_color=node_colors,
                    font_family=utils.get_config()['plot_params']['font']
                    )

        ax.set_xlim([self.axis_extension_factor*x for x in ax.get_xlim()])
        ax.set_ylim([self.axis_extension_factor*y for y in ax.get_ylim()])
        plt.tight_layout()
        plt.show()
        return ax