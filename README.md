# Car Price Prediction

## Overview

This machine learning project focuses on predicting car prices based on various features, providing a tool for estimating the fair market value of a vehicle. The project utilizes the XGBoost regression algorithm for accurate predictions.

## Files

1. **config.py:** Contains configurations for the machine learning model and data.
2. **preprocessing.py:** Handles data preprocessing steps, including feature engineering and transformation.
3. **train.py:** Implements the training of an XGBoost regression model using the provided dataset.
4. **test.py:** Evaluates the performance of the trained model using mean absolute error on a test dataset.
5. **main.py:** Orchestrates the entire machine learning pipeline, from data loading to model evaluation.

## Usage

1. **Data Loading:** Ensure the dataset ('ARM_Cars.csv') is available and contains the necessary information. You can download it from [Kaggle](https://www.kaggle.com/datasets/karenuniverse/car-sales-in-armenia-091119-041219).

2. **Preprocessing:** Run `preprocessing.py` to prepare the data for model training.

3. **Model Training:** Execute `train.py` to train the XGBoost regression model.

4. **Model Testing:** Run `test.py` to evaluate the model's performance on a separate test dataset.

5. **Main Execution:** Execute `main.py` to execute the complete machine learning pipeline.

## Dependencies

Ensure you have the required dependencies installed using the following:

```bash
pip install -r requirements.txt
