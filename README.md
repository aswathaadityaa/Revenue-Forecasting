🔍 Project Overview

This project presents a detailed comparative analysis of three forecasting models:

ARIMA (Baseline statistical model)

ARIMA-CNN (Hybrid deep learning model)

ARIMA-LSTM (Hybrid sequential model)

The objective is to identify the most accurate forecasting method for predicting quarterly business revenues that exhibit strong seasonality and non-linearity.

📘 Executive Summary

Traditional forecasting models often fail to capture complex, nonlinear and seasonal business patterns.
This study proves that hybrid models—particularly ARIMA-CNN—significantly outperform classical ARIMA.

⭐ Key Result

ARIMA-CNN achieved a 45.3% reduction in RMSE and a 30.8% reduction in MAE over baseline ARIMA.

ARIMA-LSTM showed mixed performance—better RMSE but worse MAE than ARIMA.

📂 Repository Structure
├── data/
│   ├── revenue_data.csv
├── models/
│   ├── arima.ipynb
│   ├── arima_cnn.ipynb
│   ├── arima_lstm.ipynb
├── images/
│   ├── forecast_arima.png
│   ├── forecast_cnn.png
│   ├── forecast_lstm.png
│   ├── combined_forecast.png
│   ├── performance_bar_chart.png
├── app/
│   ├── api_template.py
│   ├── model_loader.py
├── README.md

📊 Visuals
ARIMA Forecast

ARIMA-CNN Forecast

ARIMA-LSTM Forecast

Combined Model Forecast

RMSE & MAE Comparison

🔬 Methodology

The project follows the CRISP-DM Framework:

Business Understanding
Quantify forecasting capability to reduce operational and financial uncertainty.

Data Understanding
Examined quarterly revenue data containing strong trend + seasonality.

Data Preparation

Handling missing values

Stationarity transformation

Train-test split (80/20)

Residual extraction for hybrid models

Modeling

ARIMA (1,1,1)

ARIMA-CNN: 1D convolution over residual sequences

ARIMA-LSTM: sequential dependency modeling

Evaluation
Metrics used: RMSE & MAE

Deployment
Prototype API-ready architecture (FastAPI/Flask suggested)

📈 Results
Comparative Performance Table
Model	RMSE	MAE
ARIMA	68.05	48.74
ARIMA-CNN	37.12	33.65
ARIMA-LSTM	51.63	49.29

(From test dataset evaluation)

Forecast Table
Quarter	ARIMA	ARIMA-CNN	ARIMA-LSTM
2025 Q1	350.49	336.88	369.03
2025 Q2	349.61	412.80	388.03
2025 Q3	349.63	380.74	385.82
2025 Q4	349.63	332.70	370.91
2026 Q1	349.63	339.86	365.89
2026 Q2	349.63	413.47	370.83
2026 Q3	349.63	400.30	377.59
2026 Q4	349.63	320.64	373.87
🧠 Insights
Why ARIMA-CNN Wins

CNN excels at detecting local repeated seasonal patterns.

Revenue data contained strong annual seasonality, well captured by convolutional filters.

LSTM struggled because long-term dependencies were less relevant than local seasonal features.

Business Interpretation

Revenue follows a reliable seasonal cycle.

ARIMA’s flat forecast fails to capture reality → high business risk.

ARIMA-CNN offers actionable foresight for:

marketing timing

inventory planning

budgeting and capital allocation

🏢 Real-World Applications

Finance: rolling forecasts, improved capital budgeting

Marketing: targeted seasonal spending

Operations: inventory & supply chain optimization

Strategy: scenario planning & risk mitigation

🚀 Future Enhancements

Transition from univariate → multivariate forecasting

Add external features: GDP, marketing spend, competitive data

Hyperparameter tuning via Bayesian Optimization

Benchmark against Prophet, N-BEATS, Transformers

Add confidence intervals for risk-aware forecast planning

Deploy FastAPI microservice with auto-refreshing model training pipeline

🧩 Technologies Used

Python

Pandas

Statsmodels (ARIMA)

TensorFlow / Keras (CNN, LSTM)

Scikit-learn

Matplotlib

FastAPI / Flask (for deployment blueprint)

📘 How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train Models

Run Jupyter notebooks in /models/.

3️⃣ Start API (if implemented)
uvicorn app.api_template:app --reload

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you’d like to modify.

📄 License

MIT License — free to use, modify, and distribute.
