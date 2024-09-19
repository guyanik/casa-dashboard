# Casa International Workforce Prediction Dashboard

![Casa International Logo](images/logo.jpg)

## Project Overview

This Streamlit dashboard presents the results of a data science project for Casa International, an international chain of contemporary interior decoration shops. The project aims to predict the required daily workforce based on warehouse workload.

## Features

- Interactive visualizations of sales stock, work order (WO), and work list (WL) data
- Forecasting and prediction models for workforce requirements
- Daily, weekly, monthly, and yearly data aggregation options
- Correlation analysis and heatmaps

## Data Sources

The dashboard utilizes three main data sources:
1. Sales Stock
2. Sales Order (WO)
3. Sales Order (WL)

## Models

The project implements several forecasting and prediction models:
- ARIMA
- Holt-Winters
- LSTM
- Random Forest
- XGBoost

## Installation

To run this dashboard locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/guyanik/casa-dashboard.git
   ```

2. Navigate to the project directory:
   ```
   cd casa-dashboard
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

The dashboard is divided into four main tabs:
1. Sales Stock
2. WO (Work Order)
3. WL (Work List)
4. Forecast & Prediction

Navigate through these tabs to explore different aspects of the data and predictions.

## Results

- Correlation between sales orders and sales stock quantity: 82%
- Best performing models:
  - LSTM: MAE of 33 hours (Last 41 days predicted)
  - ARIMA: MAE of 33 hours (Last 14 days (7 days forecasted))

## Contributors

- H. Görkem Uyanık

## Acknowledgements

- Favicon provided by [Becris](https://www.becrisdesign.com/)
- This project was completed as part of the UHasselt Master of Statistics and Data Science program Project: Data Science course.

## Links

- [GitHub Repository](https://github.com/guyanik/casa-dashboard)
- [Dashboard Site](https://maxflow.streamlit.app/)

## License

This project is licensed under the MIT License.
