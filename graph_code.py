import networkx as nx
import numpy as np
from collections import defaultdict

def create_networkx_graph(edges, weights, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for i in range(num_nodes):
        for j, weight in weights[i].items():
            if edges[i][j] > 0:
                G.add_edge(i, j, weight=weight / edges[i][j])
    
    return G

def print_net_stats(G):
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Density: {nx.density(G):.4f}")

    # Compute connected components
    connected_components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(connected_components)}")
    print(f"Largest component size: {len(max(connected_components, key=len))}")
    conn_comp_sizes = list(map(lambda x: len(x), connected_components))
    print(f"Unique component sizes {set(conn_comp_sizes)}")

    # Compute clustering coefficient (this might be slow for large graphs)
    print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")

    # Compute degree distribution
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = defaultdict(int)
    for d in degree_sequence:
        degree_count[d] += 1

import plotly.graph_objects as go

def get_degree(G):
    degree_dict = G.degree()
    degree = torch.empty(len(degree_dict))
    for n, d in degree_dict:
        degree[n] = d
    return degree

def plot_degree_vs_freq(G, freq):
    degrees = get_degree(G)
    feats = torch.arange(len(freq))
    plt.scatter(freq, degrees, alpha=.5)
    plt.xlabel('frequency')
    plt.ylabel('degree')
    plt.show()

def plot_connected_component(G, node, max_nodes=1000):
    # Find the connected component containing the node
    component = nx.node_connected_component(G, node)
    #import pdb;
    
    # If the component is too large, sample it
    if len(component) > max_nodes:
        component = set(list(component)[:max_nodes])
    
    # Create a subgraph of the component
    subG = G.subgraph(component)
    
    # Calculate layout
    pos = nx.spring_layout(subG)
    
    # Create edges trace
    edge_x = []
    edge_y = []
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create nodes trace
    node_x = [pos[node][0] for node in subG.nodes()]
    node_y = [pos[node][1] for node in subG.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    # Color nodes by their degree
    node_adjacencies = []
    node_text = []
    for _node, adjacencies in subG.adjacency():
        node_adjacencies.append(len(adjacencies))
        #pdb.set_trace()
        node_text.append(f'Node {_node}<br># of connections: {len(adjacencies)}')
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    #pdb.set_trace()
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'Connected Component containing Node {node}',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()

    return subG

def get_distances(G, source):
    return nx.shortest_path_length(G, source, weight=None)

def custom_weight(u, v, d):
    # Assuming the weight is cosine similarity, use cosine distance for edge "length"
    return 1 - d['weight']

class FeatureGraph:
    def __init__(self, n_hidden_ae):
        self.n_hidden_ae = n_hidden_ae
        self.num_top_k = np.zeros(self.n_hidden_ae)
        self.edges = [defaultdict(int) for _ in range(self.n_hidden_ae)]
        self.weights = [defaultdict(int) for _ in range(self.n_hidden_ae)]

    @torch.no_grad()
    def create_edges_for_batch(
        self,
        tokens: Int[Tensor, "batch seq"],
        model: HookedTransformer,
        autoencoder: AutoEncoder,
        k: int,
    ) -> Float[Tensor, "(batch seq) n_hidden_ae"]:
        '''
        '''
        batch_size, seq = tokens.shape
        # (batch seq) n_hidden_ae
        acts = get_ae_feature_acts(tokens=tokens,
                                   model=model,
                                   autoencoder=autoencoder,
                                   reshape_acts=False)
        # get the indices of top k activations for all (batch * seq) tokens
        _, top_k_indices = torch.topk(torch.abs(acts), k, dim=1)
        assert top_k_indices.shape == (acts.shape[0], k), f'{ top_k_indices.shape}'
        # todo vectorize
        for token_id in range(acts.shape[0]):
            # Get the indices for the top k activations for the current token
            indices = top_k_indices[token_id]  # Shape: (k,)
            self.num_top_k[indices] += 1
            normalized_vectors = F.normalize(acts[:, indices], dim=0)  
            assert normalized_vectors.shape == (batch_size * seq, k), f'{normalized_vectors.shape}'
            # Compute the cosine similarity matrix for the selected vectors
            vect_sim = torch.matmul(normalized_vectors.t(), normalized_vectors)
            assert vect_sim.shape == (k, k), f'{vect_sim.shape}'
            for i, index_i in enumerate(indices[:-1]):
                for j in range(i+1,k):
                    node_i, node_j = index_i.item(), indices[j].item()
                    self.edges[node_i][node_j] += 1
                    self.edges[node_j][node_i] += 1
                    self.weights[node_i][node_j] += vect_sim[i][j]
                    self.weights[node_j][node_i] += vect_sim[i][j]
    
    def create_edges(self, all_tokens, batch_size, model, autoencoder, k:int = 32, log_cnt=10, 
                     process_incomplete_batch=False):
        num_seqs, seq = all_tokens.shape
        crt_idx = 0
        num_batches = num_seqs // batch_size
        for i in tqdm.trange(log_cnt+1):
            self.create_edges_for_batch(tokens=all_tokens[crt_idx:crt_idx+batch_size], 
                                        model=model, 
                                        autoencoder=autoencoder,
                                        k=k)
            if i % log_cnt == 0:
                print(f'Processed batches 0..{i}')
            crt_idx += batch_size
        if process_incomplete_batch and (crt_idx < num_seqs-1):
            self.create_edges_for_batch(tokens=all_tokens[crt_idx:], model=model, 
                                        autoencoder=autoencoder, k=k)
    
    def get_nx_graph(self):
        self.G = create_networkx_graph(self.edges, self.weights, self.num_nodes)
        return self.G
    
    def save(self, filepath):
        '''
        Save the FeatureGraph object to a file using pickle.
        '''
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"FeatureGraph object saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        '''
        Load a FeatureGraph object from a file using dill.
        '''
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        print(f"FeatureGraph object loaded from {filepath}")
        return obj