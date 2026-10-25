# Car Price Predictor

A Streamlit web app that estimates the resale price of a used car from its company, model, fuel type, manufacturing year and kilometres driven. The model is a Linear Regression pipeline trained on Quikr used-car listings (2019).

## Files

| File | Purpose |
|------|---------|
| app.py | Streamlit app (UI and prediction) |
| LinearRegressionModel.pkl | Trained model |
| Cleaned_Car_data.csv | Cleaned dataset used for the dropdown options |
| car_data_final.csv, quikr_car (1) (1).csv | Dataset files |
| car_project (1).ipynb | Notebook used for cleaning and training |
| requirements.txt | Python dependencies |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
## Notes

- The data is from 2019, so the year input is capped at 2019 and prices reflect the 2019 market.
- Model quality: hold-out R2 about 0.72, cross-validated R2 about 0.78. Individual estimates can be off by around 20% because the listings are noisy.
