# Classification Experiments for A Comparative Study of Automatic Speech Act Classification - From Logistic Regression to GPT-4o


This repository contains the code and resources for the classification experiments described in the paper:

**"A Comparative Study of Automatic Speech Act Classification - From Logistic Regression to GPT-4o"**  
[Anonymzied Authors for reviewing process]

---

## Table of Contents
- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Installation](#installation)  
- [Usage](#usage)  


---

## Overview
This repository implements the classification experiments described in the paper, including:

- Preprocessing and vectorization of the SPICE dataset
- Baseline Code
- Hyperparameter tuning on different models (Logistic Regression, Random Forest, XGBoost)
- Using GPT API

## Repository Structure
# TODO!

## Usage

1. Clone this repo:
   
```bash
git clone https://github.com/sophiakene/SpeechActClassification.git
cd repo-name
```

2. Initialize Python enrivronmentS

Importantly, note that because of conflicting package versions, two different environments are needed.
Example using conda:

** First environment for Preprocessing the data and vectorizing it **

```
conda create -n preprocessing_env
conda activate preprocessing_env
pip install -r prep_requirements.txt
```
then run preprocessing and vectorization as described under §3

** Second environment for classification experiments **


in this environment you can run 




2. Run preprocessing and vectorization scripts

Prerequisites:
- The SPICE Dataset: The data folder is expected to contain subfolders called SPICE Broadcast discussion, SPICE Broadcast interview, etc. as in the original dataset distribution.
  Each register subfolder contains two more folders for North and South Ireland, respectively, which in turn contain the annotated txt files.
- Fill in the absolute or relative path to your data folder in the second code cell (directory = "...")
- Jupyter Notebook or JupyterLab


** Preprocessing**

```
conda create -n preprocessing_env python=3.13.5
conda activate preprocessing_env
pip install -r prep_requirements.txt
```
Following that, you can run the preprocessing.ipynb Jupyter Notebook.
Output:
  - preprocessed_data.csv: A CSV file containing the preprocessed data and meta data in a tabular format
  - Speech_Acts_Distribution.png: A bar plot with speech act counts

** Vectorizing **
Importantly, note that due to version incompabilities of libraries, another environment is needed for vectorizing the preprocessed data and running the classification scripts.

```
conda deactivate
conda create -n classify_env
conda activate classify_env
pip install -r classif_requirements.txt
```

Now you can run the cells of vectorize.ipynb <br>
Input: preprocessed_data.csv (output from preprocessing <br>
Outputs: <br>
- vectorized_data.npz
- labels.npy
- filenames.npy

3.  Run Classification Experiments

- baseline.ipynb
- hyperparameter_tuning.ipynb
- xgboost_oversampling.ipynb

# TODO!

Note that for the GPT classification and fine-tuning you need an API key and insert it in the following scripts:
- 
- 










