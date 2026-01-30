Advanced Time Series Forecasting with Neural Networks and Uncertainty Quantification
 Project Overview

This project implements an advanced multivariate time series forecasting system using deep learning and uncertainty quantification techniques.
Unlike traditional point forecasting approaches, this work focuses on generating prediction intervals to quantify model uncertainty, making the forecasts more reliable for real-world decision-making.

The project uses a programmatically generated multivariate dataset and applies an LSTM-based neural network with Monte Carlo Dropout to estimate predictive uncertainty.

 Objectives

Generate and analyze a multivariate time series dataset

Build a deep learning forecasting model (LSTM)

Implement uncertainty quantification using Monte Carlo Dropout

Produce 90% prediction intervals

Evaluate model performance using both accuracy and calibration metrics

 Techniques Used

Multivariate Time Series Forecasting

Long Short-Term Memory (LSTM) Networks

Monte Carlo Dropout for Bayesian Approximation

Prediction Interval Estimation

Model Evaluation (RMSE, MAE, Coverage Probability)

 Dataset

File: multivariate_time_series_dataset (2).csv

Characteristics:

Multiple correlated time series

Trend and seasonality

Controlled noise

Target variable: Last column

Feature variables: All other columns

 Project Structure
├── multivariate_time_series_dataset (2).csv
├── lstm_uncertainty_forecasting.py
├── README.md

 Installation & Requirements

Install the required dependencies using:

pip install numpy pandas matplotlib scikit-learn tensorflow

 How to Run the Project

Clone the repository:

git clone https://github.com/your-username/advanced-time-series-uq.git


Navigate to the project folder:

cd advanced-time-series-uq


Run the program:

python lstm_uncertainty_forecasting.py

 Methodology
 Data Preprocessing

Min-Max scaling for features and target

Sliding window technique for sequence generation

80–20 train-validation split

Model Architecture

Two stacked LSTM layers

Dropout layers enabled during inference

Dense output layer for forecasting

 Uncertainty Quantification

Monte Carlo Dropout sampling

Multiple forward passes during inference

Mean prediction and variance estimation

 Prediction Intervals

90% confidence intervals computed using:

μ ± 1.64σ

 Evaluation Metrics

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

Prediction Interval Coverage Probability

These metrics assess both forecast accuracy and uncertainty calibration.

 Results

Accurate point forecasts

Well-calibrated prediction intervals

Coverage close to the expected 90% confidence level

Demonstrates robustness over noisy multivariate data

 Visualization

The project includes a visualization showing:

Actual values

Predicted mean

90% uncertainty bands

This helps interpret both predictions and associated uncertainty.

 Conclusion

This project demonstrates how deep learning models can be enhanced with uncertainty estimation to move beyond simple point forecasts.
Such approaches are critical in domains like finance, energy, healthcare, and supply chain forecasting.

 Future Enhancements

Baseline comparison with ARIMA / Prophet

Quantile Regression (P10, P50, P90)

CRPS metric implementation

Transformer-based time series model
                                                                                                          Author

RUBY
Advanced Data Science Project
