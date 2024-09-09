# Retail Demand Prediction Using Random Forest Model

This project involves developing a predictive model for estimating product demand in the retail sector using machine learning techniques. The model leverages historical sales data and applies the Random Forest algorithm to create accurate and reliable demand forecasts. This project was conducted as part of the requirements for a Bachelor of Science in Data Science at Meru University of Science and Technology.

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Project Objectives](#project-objectives)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [Limitations and Future Work](#limitations-and-future-work)
- [How to Run the Project](#how-to-run-the-project)
- [Contributors](#contributors)

## Introduction
Demand forecasting in retail is crucial for optimizing inventory management, reducing costs, and enhancing customer satisfaction. Traditional statistical methods have been supplemented by machine learning algorithms, allowing for more accurate predictions based on past sales data, market trends, and more. This project focuses on predicting product demand for an e-commerce tech-gadget retailer.

## Motivation
Retailers face challenges in balancing inventory levels with fluctuating customer demands. This project aims to address these challenges by developing a machine learning-based demand prediction model that adapts to changing market conditions.

## Project Objectives
- **General Objective**: To estimate product demand using machine learning techniques and historical sales data.
- **Specific Objectives**:
  1. Develop a predictive model for forecasting product sales.
  2. Identify key features that affect product sales volumes.
  3. Evaluate the model's accuracy and reliability.
  4. Create an interactive web application for future sales forecasts.

## Methodology
The project employs the Random Forest algorithm and time series methods like SARIMA. Data was preprocessed, and features were engineered to improve the model's performance. Hyperparameter tuning was conducted using grid search, and model validation was performed on unseen data.

## Results
The model was found to have an R-squared of 0.6755, this suggests that the model provides a reasonably good fit to the data. It indicates that a substantial portion of the variability in the target variable is captured by the features included in the model. 

## Key Findings
- **Model Performance**: The Random Forest model provided reliable demand forecasts.
- **Feature Importance**: Key features influencing demand were identified.
- **Hyperparameter Tuning**: Optimized hyperparameters significantly improved the model's accuracy.

## Recommendations
- Integrate the model into existing decision-support systems for real-time forecasting.
- Focus on key drivers of demand as identified in the feature importance analysis.

## Limitations and Future Work
- The model's performance may vary based on different data sources or time periods.
- Future research could explore alternative models or larger datasets for improved accuracy.
