import networkx as nx
from pyvis.network import Network
from collections import defaultdict
import numpy as np

class Graph:
    def __init__(self):
        self.G = nx.Graph()
        self.node_sizes = []

    def create_network_graph(self, tokenized_texts, window_size=2, min_weight=10):
        """
        Create network graph from tokenized texts
        """
        edge_weights = defaultdict(int)
        
        for text in tokenized_texts:
            for i in range(len(text)):
                start = max(0, i - window_size)
                end = min(len(text), i + window_size + 1)
                
                word1 = text[i]
                for j in range(start, end):
                    if i != j:
                        word2 = text[j]
                        edge = tuple(sorted([word1, word2]))
                        edge_weights[edge] += 1
        
        for (word1, word2), weight in edge_weights.items():
            if weight >= min_weight:
                self.G.add_edge(word1, word2, weight=weight)
        
        # Only proceed if graph has edges
        if not self.G.edges():
            print("No edges meet the minimum weight threshold. Try lowering min_weight.")
            return None
            
        self._update_node_sizes()
        
        print(f"\n📊 Graph created with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return self.G

    def _update_node_sizes(self):
        """
        Update node sizes based on degree centrality
        """
        centrality = nx.degree_centrality(self.G)
        self.node_sizes = [v * 1000 for v in centrality.values()]

    def get_max_components_num(self):
        return len(list(nx.connected_components(self.G)))

    def get_top_components(self, n_components):
        """
        Get the top N largest connected components of the graph.
        """
        components = list(nx.connected_components(self.G))
        components.sort(key=len, reverse=True)
        return components[:n_components]

    def draw_graph_pyvis(self, height="1200px", width="100%", n_components=None):
        if not self.G.edges():
            return "<p>No graph to display. The graph may be empty or all edges may have been filtered out.</p>"

        net = Network(height=height, width=width, notebook=True, cdn_resources='remote')
        
        # Get top components if specified
        if n_components is not None:
            components = self.get_top_components(n_components)
            subgraph = self.G.subgraph(set().union(*components))
        else:
            subgraph = self.G

        # Add nodes
        for node in subgraph.nodes():
            size = subgraph.degree(node)  # Use degree as size
            net.add_node(node, size=size, title=node)
        
        # Add edges
        for source, target, weight in subgraph.edges(data='weight'):
            net.add_edge(source, target, value=weight)
        
        # Set options for better visualization
        net.set_options('''
        var options = {
          "nodes": {
            "font": {
              "size": 12
            }
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": false
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 25
            }
          }
        }
        ''')
        
        # Generate and return the HTML
        return net.generate_html()

    def prune_graph(self, min_weight=2):
        original_edges = self.G.number_of_edges()
        self.G = nx.Graph((u, v, d) for (u, v, d) in self.G.edges(data=True) if d['weight'] >= min_weight)
        
        # Remove isolated nodes
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        
        self._update_node_sizes()
        
        print(f"\n🔄 Graph pruned: {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges remain")
        print(f"Removed {original_edges - self.G.number_of_edges()} edges")
        
        return self

    def get_graph_data(self):
        return {
            'nodes': list(self.G.nodes(data=True)),
            'edges': list(self.G.edges(data=True)),
            'node_sizes': self.node_sizes
        }

    @classmethod
    def from_graph_data(cls, graph_data):
        graph = cls()
        graph.G.add_nodes_from(graph_data['nodes'])
        graph.G.add_edges_from(graph_data['edges'])
        graph.node_sizes = graph_data['node_sizes']
        return graph