# GTE-PPIS-Baseline: Reproducibility Audit & Corrected Implementation

This repository contains a fully audited, scientifically reproducible baseline for the **GTE-PPIS** architecture. GTE-PPIS is a novel structure-based protein-protein interaction site predictor that uses two branches, a graph transformer and an equivariant graph neural network (EGNN), to capture amino acid information and complete site prediction tasks.

This fork serves as the corrected control benchmark for evaluating proposed joint supervision mechanisms in protein-protein interaction site (PPIS) prediction, as the original repository contained architectural discrepancies and hardcoded server paths.

## 1. System Requirements  
GTE-PPIS is developed under a Linux environment with:  
* python 3.10.14  
* numpy 1.26.4  
* pandas 2.2.2 
* scikit-learn 1.5.0   
* pytorch 2.4.1   
* dgl 2.4.0  

## 2. Dataset and Features  
The datasets used in this study (`Train_335`, `Test_60`, `Test_315-28` and `UBtest_31-6`) are stored in `./Dataset` in python dictionary format:  

    Dataset[ID] = [seq, label]

The normalized feature matrixes PSSM(L * 20), HMM(L * 20), DSSP(L * 14) and resAF(L * 7) are stored in `./Feature` in numpy format.  

Due to GitHub limitations, large data files are stored on Google Drive:
* [Distance maps (L * L)](https://drive.google.com/drive/folders/1UKX1dIrzrEPQpGyYcvIJlrm6KAKvNEpp?usp=drive_link)
* [Protein PDB files](https://drive.google.com/drive/folders/1eTjFtxsP4mnzyg5CXs96c3w4AxzYNW5N?usp=drive_link)

## 3. Baseline Selection and Reproducibility Audit
To rigorously evaluate proposed architectural improvements, an extensive audit of the official GTE-PPIS codebase was conducted. Several discrepancies between the published manuscript and the public repository were identified and systematically corrected to align this code with the paper’s reported optimal configuration:

* **Architectural Alignment:** Residual connections within the Graph Transformer (GT) block, which the original ablation study proved essential for optimal performance, were found disabled in the source code and have been re-enabled (`transformer_residual=True`). 
* **Dataset Purity:** The anomalous protein chain `2j3rA` was programmatically excluded from the training set to strictly match the 334-chain training partition (`Train_334`) described in the original study.
* **Environment Localization:** Hardcoded absolute server paths (e.g., `/pubssd/...`) were localized to ensure the pipeline can be independently executed and verified by any researcher.

## 4. Results and Discussion: The Reproducibility Gap
The published GTE-PPIS model reported an AUPRC of 0.611 and an AUROC of 0.873 on the `Test_60` dataset. However, rigorous local reproduction utilizing this corrected official codebase yielded a significantly lower average AUPRC of 0.4869 and an AUROC of 0.8229 across a 5-fold cross-validation. A subsequent independent training run confirmed this reproducible baseline ceiling, yielding an AUPRC of 0.4760 and an AUROC of 0.8146. 

This discrepancy indicates a reproducibility gap, likely stemming from unpublished post-processing steps, uncommitted hyperparameter tuning, or variations in random seed initialization in the original authors' repository.

### Performance of the Proposed Joint Supervision Model
Evaluated against this mathematically sound, locally reproducible baseline, our proposed joint RSA and B-factor supervision architecture (`ALPHA_RSA = 0.5`, `lr = 1e-4`) demonstrated a substantial improvement in predictive capability. 

Because protein-protein interaction site datasets exhibit extreme class imbalance, AUPRC serves as the most critical metric for evaluating practical model performance. The proposed architecture delivered a **+0.0754 absolute improvement in AUPRC** over the reproducible baseline. This improvement was driven by a significant increase in Precision (from 0.4329 to 0.5106), indicating that the joint supervision mechanism successfully reduced false-positive predictions without drastically sacrificing Recall.

**5-Fold Cross-Validation Performance on Test_60 Benchmark:**

| Metric | Reported GTE-PPIS (Paper) | Reproducible Baseline (This Repo) | Proposed Model (Joint Supervision) |
| :--- | :--- | :--- | :--- |
| **AUPRC** | 0.611 | 0.4869 | **0.5623** |
| **AUROC** | 0.873 | 0.8229 | **0.8569** |
| **MCC** | 0.500 | 0.4008 | **0.4525** |
| **F1-Score** | 0.582 | 0.5042 | **0.5420** |
| **Precision**| 0.557 | 0.4329 | **0.5106** |
| **Recall** | 0.611 | 0.6041 | 0.5831 |

## 5. Running GTE-PPIS 
Train the baseline model with the corrected, paper-aligned default parameters:  

    python train.py
 
*(Training will automatically generate a timestamped folder inside the `./Log/` directory).*

Test the model you just trained on the test sets by passing the generated log directory as an argument:  

    python test.py --log_dir <TIMESTAMP_FOLDER_NAME>

You can adjust the architectural parameters via `EGNN_model.py` and `GraphTransformer_Block.py`.
