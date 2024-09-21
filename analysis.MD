# Analysis of Sparse Autoencoder Features with Co-Activation Graphs
## Goal

## Minimal SAE background and motivation 

Autoencoders are neural network architectures used to learn efficient representations (encodings) of data, typically in an unsupervised manner.

As the goal of Mechanistic Interpretability is to "reverse-engineer" the computations performed by a neural network, having a clean mapping between neurons and human-interpretable concepts would greatly facilitate our understanding of the model's internal processes.
Yet, as shown by works such as [Toy models of superposition](https://transformer-circuits.pub/2022/toy_model/index.html), [Finding Neurons In A Haystack](https://arxiv.org/pdf/2305.01610), LLMs superimpose neurons in order to represent as many different concepts as useful, resulting in polysemantic neurons.

By mapping the neuron activations to a larger space $\mathbb{R}^{d_{\text{ae\_hidden}}}$, $d_{\text{ae\_hidden}} > d_{\text{mlp}}$ and encouraging sparsity, the hope is that each dimension in $\mathbb{R}^{d_{\text{ae\_hidden}}}$ corresponds to human interpretable features. This mapping is achieved by training a sparse autoencoder on the regularized reconstruction loss.

### Problem Setup
Let $x \in \mathbb{R}^{d_{\text{mlp}}}$ be the MLP activations. 

The autoencoder consists of:

- **Encoder**: An encoding matrix $W_{\text{enc}} \in \mathbb{R}^{d_{\text{mlp}} \times d_{\text{ae\_hidden}}}$ that maps the neuron activations $x$ (containing polysemantic features) into a higher-dimensional space:

  $$ z = ReLU((x-b_{\text{dec}}) W_{\text{enc}} + b_{\text{enc}}) $$

- **Decoder**: A decoding matrix $W_{\text{dec}} \in \mathbb{R}^{d_{\text{ae\_hidden}} \times d_{\text{mlp}}}$ that maps the features back to the MLP space:

  $$ \hat{x} = z W_{\text{dec}} + b_{\text{dec}} $$

The autoencoder is trained on the regularized reconstruction loss.

The typical workflow involves:

1. **Data Preparation**: Convert input text to tokens and pass them through the transformer model, recording the MLP activations for a specified layer.

2. **Training the Autoencoder**: Use these activations to train the sparse autoencoder.

3. **Feature Extraction**: The encoded representations $z$ (SAE features) are used for interpretation and analysis.


## Analysis

### Creating the feature graph
A co-activations graph is defined as follows:

- **Nodes**: Each node represents an SAE feature (i.e., one of the $d_{\text{ae\_hidden}}$ neurons in the encoding layer of the SAE).

- **Edges**: For every token in the training dataset, we identify the top $k$ most active SAE features (those with the highest activation values). We connect these features with edges, where the edge weight is the average cosine similarity of their activations across all tokens in the batch.

In this analysis, the graphs are computed over the first 50 training data batches using $k=4$, i.e., the top 4 most active features.

#### Feature Degree vs Frequency
![alt text](degree_dist.png)
![alt text](log_deg_dist.png)
![alt text](freq_dist.png)

The distributions of feature degree and frequency are bimodal. The Pearson correlation coefficient between the two distributions is approximately 0.51, indicating a moderate positive correlation. Notably, there is a non-negligible number of features with either low degree and high frequency or vice versa. Likely, the features with low frequency co-activate with a diverse set of features while features with the converse for high frequency and low degree.

![alt text](feat_deg_vs_frq.png)

#### Descriptive statistics

Graph for SAE1:

`Number of nodes: 16384`

`Number of edges: 794239`

`Average degree: 96.95`

`Density: 0.0059`

`Number of connected components: 9411`

`Largest component size: 6974`

`Unique component sizes {1, 6974}`

`Average clustering coefficient: 0.1379`


Graph for SAE2:

`Number of nodes: 16384`

`Number of edges: 795547`

`Average degree: 97.11`

`Density: 0.0059`

`Number of connected components: 9434`

`Largest component size: 6951`

`Unique component sizes {1, 6951}`

`Average clustering coefficient: 0.1373`


#### Usage example

