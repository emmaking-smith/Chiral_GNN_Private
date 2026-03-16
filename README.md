# Using Graph Neural Networks to Predict Absolute Stereochemical Configuration

## Overview
This repository contains the code, datasetfor a BSc research project on predicting the absolue configuration via optical rotation labels of chiral molecules using classical machine learning and graph neural network models. The project compares chirality-aware Morgan fingerprints and graph-based molecular representations, and includes ablation studies to evaluate the contribution of different embedded node features.

## Methods overview
This project compares four classical machine learning (ML) models and five graph neural network (GNN) models for binary classification of optical rotation labels.

### Classical ML models
- Random Forest
- Extra Trees
- Gradient Boosting
- Support Vector Machine (SVM)
The detailed implementation of the ML models can be found in benchmark.py

These models use chirality-aware Morgan fingerprints as input features, the generation of the hirality-aware Morgan fingerprints can be found in dataconversion.py and smiles_to_moreganfingerprint.py

### GNN models
- Graph Convolutional Network (GCN)
- Graph Isomorphism Network (GIN)
- Graph Attention Network (GAT)
- Graph Sample and Aggregate (GraphSAGE)
- Attentive Fingerprint (AttentiveFP)

For the GNN models, each molecule is represented as a molecular graph in which atoms are treated as nodes and bonds as edges. Node features include atomic number, chirality type, hybridisation, and atomic coordinates. Ablation studies were carried out to examine the effect of removing selected node features.

All models were evaluated using five-fold cross-validation with a fixed random state for consistency across methods.
The implementation of the GNNs can be found in run_model.py and torch_geometric_model_loading.py
The generation of the molecular graph can be found in the smiles_to_geometric_data.py

## Dataset
the analysis code of dataset was named as dataset_analysis.py

## Result Analysis
The GNN_analysis_code.py and run_GNN_analysis.py contribute to the analysis of all GNN experiments.
The benchmark.analysis.py contributes to the benchmark result analysis.
comparison_plots.py helps plot the scatter plot of the ablation experoment.

## Requirements
- Python 3.10.15
- PyTorch 2.5.1
- PyTorch Geometric
- scikit-learn 1.7.2
- RDKit
- NumPy
- pandas
- matplotlib

