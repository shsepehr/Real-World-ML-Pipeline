# Real-World ML Pipeline

**End-to-end production-ready Python machine learning pipeline for data preprocessing, feature engineering, model training, and predictions.**

> ⚠️ Note: The dataset used in this project is a sample CSV. You should replace it with your real data when applying the pipeline.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project demonstrates a **real-world ML pipeline** that handles:

1. **Data preprocessing**: missing values imputation, scaling, and categorical encoding.
2. **Feature engineering**: creating new features and transformations.
3. **Model training**: training a logistic regression model with scikit-learn pipeline.
4. **Prediction & output**: generating predictions for new data, safely handling unseen categories.
5. **Model persistence**: saving and loading trained models using `joblib`.

The pipeline is designed to be **production-ready** and modular, making it easy to integrate into real-world applications.

---

## Features

- Modular Python code with `features.py`, `preprocess.py`, `train.py`, and `predict.py`
- Handles **unseen categorical values** in test/new data using `OneHotEncoder(handle_unknown="ignore")`
- Separate training and prediction scripts
- End-to-end pipeline from raw CSV input to predictions output
- Easy to extend with other ML models or feature engineering

---

## Project Structure

