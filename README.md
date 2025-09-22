# Classification Experiments for A Comparative Study of Automatic Speech Act Classification - From Logistic Regression to GPT-4o


This repository contains the code and resources for the classification experiments described in the paper:

**"A Comparative Study of Automatic Speech Act Classification - From Logistic Regression to GPT-4o"**  
[Anonymzied Authors for reviewing process]

---

## Table of Contents
- [Overview](#overview)  
- [Usage](#usage)  


---

## Overview
This repository implements the classification experiments described in the paper, including:

- Preprocessing and vectorization of the SPICE dataset
- Baseline Code
- Hyperparameter tuning on different models (Logistic Regression, Random Forest, XGBoost)
- Using GPT API


## Usage

1. Clone this repo:
   
```bash
git clone https://github.com/sophiakene/SpeechActClassification.git
cd repo-name
```

2. Preprocessing

Importantly, note that because of conflicting package versions, two different environments are needed.
Example using conda:

** First environment for Preprocessing the data **

```
conda create -n preprocessing_env
conda activate preprocessing_env
pip install -r prep_requirements.txt
```
then run the cells in preprocessing.ipynb



2. **Vectorizing**

Prerequisites:
- The SPICE Dataset: The data folder is expected to contain subfolders called SPICE Broadcast discussion, SPICE Broadcast interview, etc. as in the original dataset distribution.
  Each register subfolder contains two more folders for North and South Ireland, respectively, which in turn contain the annotated txt files.
- Fill in the absolute or relative path to your data folder in the second code cell (directory = "...")
- Jupyter Notebook or JupyterLab

Importantly, note that due to version incompabilities of libraries, another environment is needed for vectorizing the preprocessed data and running the classification scripts.

** Second environment for vectorization and classification experiments **

```
conda create -n preprocessing_env python=3.13.5
conda activate preprocessing_env
pip install -r prep_requirements.txt
```
Following that, you can run the preprocessing.ipynb Jupyter Notebook.
Output:
  - preprocessed_data.csv: A CSV file containing the preprocessed data and meta data in a tabular format
  - Speech_Acts_Distribution.png: A bar plot with speech act counts

3. **Classifying**

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


Note that for the GPT classification and fine-tuning you need an API key and insert it in the following script:
- zero-and-few-shot-clf.ipynb

The notebook gpt_fine_tuning.ipynb prepares the prompts and train, validate and test set for fine-tuning GPT-4o. The actual fine-tuning was carried out on platform.openai.com.










