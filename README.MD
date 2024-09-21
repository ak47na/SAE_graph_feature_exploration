# Exploring Sparse Autoencoder Features with Co-Activation Graphs

## Goal and Motivation

Following the recent success in scaling SAEs, several works (such as [Open Source Automated Interpretability for Sparse Autoencoder Features](https://blog.eleuther.ai/autointerp/) and [Disentangling Dense Embeddings with Sparse Autoencoders](https://arxiv.org/html/2408.00657v1#S4)) have focused on facilitating or automating feature discovery and evaluation.

Inspired by their findings on the geometric structure of features, and the existence of universal features (i.e. features with similar activation patterns across different SAEs), this project builds a co-activation graph and uses its structure to reason about features (and feature groups) within and across SAEs.

In contrast to prior work, features from different autoencoders belong to distinct graphs, which requires less computation. This approach allows for comparing their structures and facilitates both local and global understanding of features.

The broader aims are:
1. test the feature universality hypothesis by comparing graph structures 
2. facilitate circuit analysis by looking at paths inside the graph
3. use the graph as a simplified/compressed map of the model's knowledge representations 
4. understand the differences in SAE features between standard models and safety fine-tuned models

Although the graph constructed in this work is a simplified representation, adopting a network science perspective may offer insights that analyzing features individually may provide.

For instance, this project tackles the following questions: 
1. Do autoencoders trained on the same dataset under the same training procedure but different seeds have similar co-activation graph structures?
2. Can node statistics in one graph facilitate finding nodes corresponding to related features in another?
3. In what ways does information about one feature assist in the identification of related features?"


## Setup

At a high level, SAEs are used to identify meaningful representations (e.g. monosemantic features) from the activations of a neural network layer. $^1$ For a detailed explanation of SAEs, please see ARENA's Transformer Interpretability Chapter, section [1.3.1](https://arena3-chapter1-transformer-interp.streamlit.app/[1.3.1]_Superposition_&_SAEs), or the `analysis.MD` file for the *minimal* background. 

### Model and Data

The SAEs are trained on the MLP activations of the `gelu-1l` model in TransformerLens on Neel Nanda's `c4-code-20k` dataset (10K elements of C4 and 10K elements of code parrot clean (Python code)).
The trained SAEs are taken from HuggingFace from [NeelNanda/sparse_autoencoder](https://huggingface.co/NeelNanda/sparse_autoencoder), and the associated [analysis notebook](https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn?usp=sharing) is used as a starting point for this project.

## Results

For the methodology of creating the co-activation graphs and additional results, please see `analysis.MD` or `SAE_feature_graph_analysis.ipynb` for reproducing the experiments.

The graph structure (degree and strength distribution) for SAEs of the same model trained with different seeds is very similar (see plots below). In particular, the Jensen-Shannon divergence between the degree distributions of the two graphs (G1, G2) is 0.15 (a random baseline with the same number of nodes and edges gives a divergence of ~.73)

[!alt text][degree_dist.png]

A naive search based on the graph structure allows finding nodes in G2 that are similar to a node in G1, with the similarity carrying over to the "SAE feature space". For example, one of the top similar nodes in G2 to node #16124 in G1 (which as a feature activates highly on tokens such as "un", "iversity", "rule" and boosts logits such as "student", "college") is #6741 (which activates highly on tokens such as "Clear", "information", "universal" and boosts logits such as "correctly", "contents"). 

While these results support the first two research questions, future work is required to make the analysis more robust.


## Future work
1. Can the graph structure increase our understanding of the AI's world model (for instance by identifying communities of features in the graph)?
2. Are nodes with low frequency and high degree (or vice-versa) meaningful? How to assess their value/interpretability?
3. What other methods for linking features and associating weights are more appropriate for feature discovery?
4. Can the paths between features provide relevant information about the model's internal
representations and "beliefs"? Can one leverage graph traversals to interpret chain of thought traces?

## References
[Disentangling Dense Embeddings with Sparse Autoencoders](https://arxiv.org/html/2408.00657v1#S4)
[Sparse Autoencoder.ipynb](https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn?usp=sharing)
[Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/)



## Interesting resources
[Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/html/2403.19647v1)
[Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs](https://www.lesswrong.com/posts/G2oyFQFTE5eGEas6m/interpretability-as-compression-reconsidering-sae)

## Notes

$1.$ SAEs are more general, and the encodings/representations are typically extracted from unlabelled data (the neuron activations in this setup)
