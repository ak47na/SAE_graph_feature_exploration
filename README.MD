# SAE Feature Discovery using co-activation graphs

## Goal and Motivation

Following the recent success in scaling SAEs, several works (such as [Open Source Automated Interpretability for Sparse Autoencoder Features](https://blog.eleuther.ai/autointerp/) and [Disentangling Dense Embeddings with Sparse Autoencoders](https://arxiv.org/html/2408.00657v1#S4)) have focused on facilitating or automating feature discovery and evaluation.

Inspired by their findings on the geometric structure of features, and the existance of universal features (i.e. features with similar activation patterns across different SAEs), this project builds a co-activation graph and uses its structure to reason about feature groups within and across SAEs.

In particular, questions of interest are:
1. Is there a principled way of finding features? 
2. How does information about one feature facilitate identifying another?
3. How do SAE features differ across standard and safety fine-tuned models?

## Setup

At a high level, SAEs are used to identify meaningful representations (e.g. monosemantic features) from the activations of a neural network layer.$[^1]$ For a detailed explanation of SAEs, please see ARENA's Transformer Interpretability Chapter, section [1.3.1](https://arena3-chapter1-transformer-interp.streamlit.app/[1.3.1]_Superposition_&_SAEs), or the `analysis.MD` file for the *minimal* background. 

### Model and Data

The SAEs are trained on the MLP activations of the `gelu-1l` model in TransformerLens on Neel Nanda's `c4-code-20k` dataset (10K elements of C4 and 10K elements of code parrot clean (Python code)).
The trained SAEs is taken from HuggingFace from [NeelNanda/sparse_autoencoder](https://huggingface.co/NeelNanda/sparse_autoencoder), and the associated [analysis notebook](https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn?usp=sharing) is used as a starting point for this project.

## Current work

The graph structure (degree and strength distribution) for SAEs of the same model trained with different seeds is very similar (see plots below). In particular, the Jensen-Shannon diveregence between the degree distributions of the two graphs (G1, G2) is 0.15 (a random baseline with the same number of nodes and edges gives a divergence of ~.73)

A naive search based on the graph strcuture allows finiding nodes in G2 that are similar to a node in G1, with the similarity carrying over to the "SAE feature space". For example, one of the top similar nodes in G2 to node #16124 in G1 (which as a feature activates highly on tokens such as "un", "iversity", "rule" and boosts logits such as "student", "college") is #6741 (which activates highly on tokens such as "Clear", "information", "universal" and boosts logits such as "correctly", "contents"). 

For general results on the feature frequency distributions and descriptive statistics over the co-activations graph, please see `analysis.MD`.


## Possible experiments

2. Test if autoencoders trained on the different models and/or different datasets have similar graph structures
3. Study communites of features


## Future work
1. Can the graph structure increase our understanding of the AI's world model?
2. Are nodes with low frequency and high degree (or vice-versa) meaningful? How to assess their value/interpretability?
3. What other methods for linking features and associating weights are more appropriate for
feature discovery?
4. Can the paths between features provide relevant information about the model's internal
representations and "beliefs"? Can one leverage graph traversals to interpret chain of thought traces?

## References


## Interesting resources
(https://blog.eleuther.ai/autointerp/)
(https://arxiv.org/html/2403.19647v1)
(https://www.lesswrong.com/posts/G2oyFQFTE5eGEas6m/interpretability-as-compression-reconsidering-sae)

## Notes

$[^1]$ SAEs are more general, and the encodings/representations are typically extracted from unlabelled data (the neuron activations in this setup)
