# Question Pair Detection: ML/DL Model

This project focuses on building both **machine learning (ML)** and **deep learning (DL)** models to detect whether two questions from **Quora** are duplicates. We use the **Quora Question Pairs** dataset, which contains pairs of questions with labels indicating whether the pair is a duplicate or not.

We explore multiple approaches, including traditional ML methods like **Logistic Regression** and **XGBoost Classifier**, and more advanced **Deep Learning (DL)** methods using **BERT** for fine-tuned text classification.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Analysis](#analysis)
  - [Duplicates vs Non-Duplicates Distribution](#duplicates-vs-non-duplicates-distribution)
  - [Text Length Distribution](#text-length-distribution)
- [Models](#models)
  - [Logistic Regression](#logistic-regression)
  - [XGBoost Classifier](#xgboost-classifier)
  - [BERT (Deep Learning)](#bert-deep-learning)

## Overview

The goal of this project is to create a model that can classify whether two questions on Quora are duplicates. The dataset provides pairs of questions with a binary label (`1` for duplicates, `0` for non-duplicates). The project involves data preprocessing, feature extraction, model training, and performance evaluation.

We use the following approaches:
- **Logistic Regression**: A basic machine learning model suitable for binary classification.
- **XGBoost Classifier**: A powerful gradient-boosted decision tree model that excels in tabular data classification tasks.
- **BERT (Bidirectional Encoder Representations from Transformers)**: A state-of-the-art deep learning model pre-trained on vast amounts of text and fine-tuned for this specific task.

## Dataset

The dataset used for this project is the **Quora Question Pairs** dataset, which is publicly available. The dataset consists of pairs of questions and a binary label indicating whether the questions are duplicates (1) or not (0).

- **Dataset Source**: [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs/data)
- **Columns**:
  - `question1`: The first question in the pair.
  - `question2`: The second question in the pair.
  - `is_duplicate`: Label indicating if the questions are duplicates (`1` for duplicate, `0` for non-duplicate).

## Analysis

Before building the models, we conducted some exploratory data analysis (EDA) to better understand the dataset:

### Duplicates vs Non-Duplicates Distribution
We analyzed the distribution of duplicate vs. non-duplicate pairs using a histogram. This analysis helps us understand the class imbalance in the dataset.

### Text Length Distribution
We also explored the distribution of text lengths for both `question1` and `question2`, as longer questions might have different characteristics compared to shorter ones.

## Models

### Logistic Regression

We used **Logistic Regression** as one of the traditional machine learning algorithms for binary classification. This model is a linear classifier and works well for relatively simple, linearly separable tasks.

- **Preprocessing**: TF-IDF Vectorization was used to convert text data into numerical features.
- **Evaluation**: The model's performance was evaluated using accuracy, precision, recall, and F1-score.

### XGBoost Classifier

**XGBoost** (Extreme Gradient Boosting) is a highly effective gradient boosting algorithm. We used XGBoost to train a classifier on the TF-IDF features of the question pairs.

- **Preprocessing**: TF-IDF Vectorization.
- **Evaluation**: Similar evaluation metrics as Logistic Regression, but XGBoost typically performs better due to its ensemble learning approach.

### BERT (Deep Learning)

**BERT** (Bidirectional Encoder Representations from Transformers) is a state-of-the-art deep learning model designed for natural language understanding tasks. We fine-tuned BERT for the task of question pair classification.

- **Preprocessing**: Tokenization using BERT's tokenizer.
- **Evaluation**: Performance was evaluated on the validation set, providing insight into the model's ability to handle complex textual relationships.

