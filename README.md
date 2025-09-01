# MGLTU-Net: Multi-Representation Graph Learning with Triplet Cross-Attentive Uncertainty-Guided Network
## Abstract  

Effective disaster response relies on timely and accurate damage assessment, yet existing methods struggle with ambiguous visual cues, heterogeneous damage patterns, and modality-specific uncertainties.  
In this work, we propose a principled framework that unifies **multi-representation graph learning**, **triplet cross-attention**, and **uncertainty-guided fusion** to address these challenges.  
Our approach introduces:  
1. A **Semantic Class-Aware Graph module** for semantically grounded relational reasoning.  
2. A **Multi-Scale Triplet Cross-Attention mechanism** to capture spatial dependencies across multiple dimensions.  
3. An **Uncertainty-Weighted Fusion strategy** that ensures robust integration even under compromised modalities.  

To support further research, we also introduce **DAMAGE**, a large-scale, manually annotated benchmark for complex damage assessment.


## Project Structure: MGLTC-Net

The repository is organized as follows:

## Project Structure

This repository contains the implementation of **MGLTC-Net: Multi-Representation Graph Learning with Triplet Cross-Attention and Uncertainty-Guided Fusion for Disaster Damage Assessment**.

```bash
MGLTC-Net/
│
├── data/                  
│   ├── crisismmd/
│   └── damage/
│
├── models/                
│   ├── mgltc_net.py       
│   ├── semantic_graph.py  
│   ├── triplet_attention.py
│   └── fusion.py          
│
├── utils/                 
│   ├── metrics.py
│   ├── losses.py
│   ├── visualization.py
│   └── explainability.py
│
├── train.py               
├── evaluate.py            
├── inference.py           
│
├── configs/               
│   └── default.yaml       
│
├── requirements.txt       
├── README.md

```

##Methodology

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
## Requirements  

To set up the environment for **MGLTC-Net**, install the following dependencies:  

### Core Dependencies  
- **torch>=2.0.0** – Deep learning framework for model training and inference  
- **torchvision>=0.15.0** – Computer vision utilities and datasets  
- **timm>=0.9.0** – Access to advanced backbones (e.g., Swin Transformer, ViT)  
- **transformers>=4.30.0** – Pretrained language/vision models from HuggingFace  
- **pytorch-lightning>=2.0.0** – High-level training loop abstraction (optional but recommended)  

### Scientific & Utility Libraries  
- **numpy>=1.23.0** – Numerical computations  
- **scipy>=1.10.0** – Scientific computing utilities  
- **scikit-learn>=1.2.0** – Metrics, preprocessing, and ML utilities  
- **pandas>=2.0.0** – Data manipulation and analysis  
- **tqdm>=4.65.0** – Progress bar for training and evaluation  

### Graph Learning  
- **spektral>=1.3.0** – Graph neural network library (core for our relational reasoning)  
- **einops>=0.6.0** – Flexible tensor operations for attention mechanisms  
- **torchmetrics>=0.11.0** – Standardized evaluation metrics  

### Visualization & Augmentation  
- **matplotlib>=3.7.0** – Visualization of results and plots  
- **seaborn>=0.12.2** – Statistical data visualization  
- **opencv-python>=4.7.0** – Image processing  
- **albumentations>=1.3.0** – Advanced data augmentation  

### Explainability (Optional but Recommended)  
- **captum>=0.6.0** – Model interpretability (saliency maps, occlusion, gradients)  
- **shap>=0.41.0** – SHAP-based explainability methods  

---

### Installation  

Clone the repository and install requirements:  

```bash
git clone https://github.com/Shahid-135/MGLTC-Net.git
cd MGLTC-Net
pip install -r requirements.txt



