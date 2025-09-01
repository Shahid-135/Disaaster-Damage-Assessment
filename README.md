# MGLTU-Net: Multi-Representation Graph Learning with Triplet Cross-Attentive Uncertainty-Guided Network
## Abstract  

Effective disaster response relies on timely and accurate damage assessment, yet existing methods struggle with ambiguous visual cues, heterogeneous damage patterns, and modality-specific uncertainties.  
In this work, we propose a principled framework that unifies **multi-representation graph learning**, **triplet cross-attention**, and **uncertainty-guided fusion** to address these challenges.  
Our approach introduces:  
1. A **Semantic Class-Aware Graph module** for semantically grounded relational reasoning.  
2. A **Multi-Scale Triplet Cross-Attention mechanism** to capture spatial dependencies across multiple dimensions.  
3. An **Uncertainty-Weighted Fusion strategy** that ensures robust integration even under compromised modalities.  

To support further research, we also introduce **DAMAGE**, a large-scale, manually annotated benchmark for complex damage assessment.

# Methodology

We propose a **multi-representation framework** for disaster damage severity classification that integrates **semantic graph reasoning, multi-scale visual attention, cross-modal alignment, uncertainty-guided fusion, and explainable features**.

---

## 1. Semantic Class-Aware Graph Adjacency Refinement (SCAR)

- Construct a **class-aware graph** using cosine similarity between pooled class prototypes.  
- Refine adjacency by **reinforcing intra-class edges** and attenuating inter-class ones.  
- Apply **GraphSAGE** for message passing to capture context-aware structural dependencies.  
- Produces **discriminative, semantically smooth embeddings** useful for fine-grained classification.

---

## 2. Feature Extraction

We adopt the **Swin Transformer** backbone to extract:

- **Global representation**: pooled feature vector `x_i ∈ ℝ^F`.  
- **Local representation**: patch-level features `H_i ∈ ℝ^(P × F)`.

These serve as inputs for **graph modeling** and **attention modules**.

---

## 3. Multi-Scale Triplet Cross-Attention (MSTCA)

MSTCA models dependencies along three complementary dimensions:

- **Channel Attention (CA):** highlights semantic feature importance.  
- **Height Attention (HA):** captures vertical spatial dependencies.  
- **Width Attention (WA):** captures horizontal spatial dependencies.  

The outputs are fused via **adaptive softmax weighting** with global context modulation.  
A **cross-dimensional self-attention** refines the joint representation, enhancing robustness to visual ambiguity.

---

## 4. Cross-Modal Graph Refinement (CMGR)

- Project visual embeddings into graph space.  
- Use **visual embeddings as queries** in a multi-head attention mechanism over graph embeddings.  
- Fuse refined and original features via a **learnable gating mechanism**.  
- Result: **visually grounded graph features** that bridge semantic reasoning with visual cues.

---

## 5. Uncertainty-Weighted Fusion (UWF)

- Map both **graph** and **visual** features into a shared latent space.  
- Estimate **epistemic uncertainty** via *Monte Carlo dropout*.  
- Compute **adaptive fusion weights**: modalities with lower uncertainty receive higher weights.  
- Ensures stable and reliable performance even when one modality is compromised.

---

## 6. Explainable Features

We enhance interpretability by incorporating:

- **Saliency maps**  
- **Occlusion sensitivity analysis**  
- **Gradient-based activations**  
- **Structural descriptors**

These explainable features provide **transparent justifications** for model predictions — critical for trust and adoption in disaster response.

---


