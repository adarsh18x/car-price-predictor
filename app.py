import pickle

import numpy as np
import pandas as pd
import streamlit as st

CURRENT_YEAR = 2019
LUXURY = {'Audi', 'Mercedes', 'BMW', 'Jaguar', 'Land', 'Mini', 'Volvo', 'Porsche'}
CATS = ['company', 'fuel_type', 'brand_model']
NUMS = ['age', 'log_kms', 'kms_per_year', 'luxury', 'age_x_lux', 'dep',
        'dep_x_lux', 'kms_bin', 'low_kms', 'age2', 'age_x_kms']


# named functions required to unpickle the model's TransformedTargetRegressor
def price_transform(y):
    return np.power(y, 0.4)


def price_inverse_transform(y):
    return np.power(np.clip(y, 0, None), 2.5)


def add_features(df):
    df = df.copy()
    df['age'] = CURRENT_YEAR - df['year']
    df['log_kms'] = np.log1p(df['kms_driven'])
    df['kms_per_year'] = df['kms_driven'] / df['age'].clip(lower=1)
    df['brand_model'] = df['name'].str.split().str[:2].str.join(' ')
    df['luxury'] = df['company'].isin(LUXURY).astype(int)
    df['age_x_lux'] = df['age'] * df['luxury']
    df['dep'] = np.exp(-0.12 * df['age'])
    df['dep_x_lux'] = df['dep'] * df['luxury']
    df['kms_bin'] = pd.cut(df['kms_driven'], [-1, 20000, 40000, 60000, 90000, 1e7], labels=False)
    df['low_kms'] = (df['kms_driven'] < 5000).astype(int)
    df['age2'] = df['age'] ** 2
    df['age_x_kms'] = df['age'] * df['log_kms']
    return df


@st.cache_resource
def load_model():
    with open('LinearRegressionModel.pkl', 'rb') as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv('Cleaned_Car_data.csv', index_col=0)


model = load_model()
car_data = load_data()

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Predictor")
st.caption("Estimate the resale price of a used car based on 2019 Quikr listings.")

company = st.selectbox("Company", sorted(car_data['company'].unique()))

matching = car_data[car_data['company'] == company]
name = st.selectbox("Model", sorted(matching['name'].unique()))
fuel_type = st.selectbox("Fuel type", sorted(matching['fuel_type'].unique()))

col1, col2 = st.columns(2)
with col1:
    # model was trained on 2019 listings; later years give negative age and wrong prices
    year = st.number_input("Manufacturing year", min_value=1995, max_value=CURRENT_YEAR,
                           value=2015, step=1)
with col2:
    kms_driven = st.number_input("Kilometers driven", min_value=0, max_value=400000,
                                 value=30000, step=1000)

if st.button("Predict Price", type="primary"):
    row = pd.DataFrame([dict(name=name, company=company, year=year,
                              kms_driven=kms_driven, fuel_type=fuel_type)])
    features = add_features(row)[CATS + NUMS]
    price = model.predict(features)[0]
    st.success(f"Estimated Price: ₹ {price:,.0f}")
    st.caption("Estimates are approximate (model R² ≈ 0.72) and reflect 2019 market prices.")
