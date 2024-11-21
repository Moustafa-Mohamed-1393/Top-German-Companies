import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import streamlit as st
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(layout='wide')

# Load the dataset
data = pd.read_csv("./Top_12_German_Companies NEW.csv")

# Page Header
st.title("German Companies Financial Data Analysis")

# Subheader for data loading
st.subheader("Loading Data")
st.write(data.head())

# Descriptive Statistics
st.write("### Descriptive Statistics")
st.dataframe(data.describe().T)

# Exploratory Data Analysis (EDA)
st.write("### EDA")
st.write("Perform your exploratory visualizations and insights here.")
st.write("### Exploratory Data Analysis")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select numeric columns
if not numeric_data.empty:
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = numeric_data.corr()  # Compute correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', square=True, ax=ax, cmap="coolwarm")
    plt.title('Correlation Heatmap')
    st.pyplot(fig)
else:
    st.write("No numeric data available to compute the correlation matrix.")

st.title("Distribution Analysis of Financial Metrics")

# Column selection
columns_to_plot = ['Revenue', 'Net Income', 'Assets', 'Liabilities', 'Debt to Equity', 'ROA (%)', 'ROE (%)']
st.sidebar.header("Select Columns to Plot")
selected_columns = st.sidebar.multiselect("Choose columns to display distributions:", columns_to_plot, default=columns_to_plot)

# Plot distributions for selected columns
st.header("Distributions of Selected Columns")
if selected_columns:
    for col in selected_columns:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data[col].dropna(), kde=True, bins=30, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
else:
    st.write("Please select at least one column to display.")
# Page Header
st.title("Financial Insights of Top German Companies")

# Total Revenue by Company
st.subheader("Total Revenue by Company")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=data, x='Company', y='Revenue', estimator=sum, ci=None, ax=ax)
ax.set_title("Total Revenue by Company")
ax.set_ylabel("Revenue")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# Total Net Income by Company
st.subheader("Total Net Income by Company")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=data, x='Company', y='Net Income', estimator=sum, ci=None, ax=ax)
ax.set_title("Total Net Income by Company")
ax.set_ylabel("Net Income")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
st.pyplot(fig)

# Total Assets of Top Companies
st.subheader("Total Assets of Top Companies")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=data, x='Company', y='Assets', estimator=sum, ci=None, ax=ax)
ax.set_title("Total Assets of Top Companies")
ax.set_ylabel("Assets")
ax.set_xlabel("Company")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(True)
st.pyplot(fig)

# Total Liabilities of Top Companies
st.subheader("Total Liabilities of Top Companies")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=data, x='Company', y='Liabilities', estimator=sum, ci=None, ax=ax)
ax.set_title("Total Liabilities of Top Companies")
ax.set_ylabel("Liabilities")
ax.set_xlabel("Company")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(True)
st.pyplot(fig)

# Scatter Plot: Assets vs. Liabilities
st.subheader("Assets vs. Liabilities")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=data, x='Assets', y='Liabilities', hue='Company', ax=ax)
ax.set_title("Assets vs. Liabilities")
ax.set_xlabel("Assets")
ax.set_ylabel("Liabilities")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig)
# Convert the Period column to datetime format
data['Period'] = pd.to_datetime(data['Period'], format='%m/%d/%Y')

# Time Series Analysis
# Aggregate metrics over time (sum Revenue and Net Income by Period)
time_series_data = data.groupby('Period').agg({
    'Revenue': 'sum',
    'Net Income': 'sum',
    'Liabilities': 'sum',
    'Assets': 'sum'
}).reset_index()
# Page Header
# Page Header
st.title("Company-Specific Financial Trends Over Time")

# Revenue Trends by Company
st.subheader("Revenue Trends by Company")
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=data, x='Period', y='Revenue', hue='Company', marker='o', ax=ax)
ax.set_title("Revenue Trends by Company")
ax.set_xlabel("Time")
ax.set_ylabel("Revenue")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
st.pyplot(fig)

# Net Income Trends by Company
st.subheader("Net Income Trends by Company")
fig, ax = plt.subplots(figsize=(14, 10))
sns.lineplot(data=data, x='Period', y='Net Income', hue='Company', marker='o', ax=ax)
ax.set_title("Net Income Trends by Company")
ax.set_xlabel("Time")
ax.set_ylabel("Net Income")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
st.pyplot(fig)

# Time-Series Plot for Revenue and Net Income
st.subheader("Revenue and Net Income Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=data, x='Period', y='Revenue', label='Revenue', marker='o', ax=ax)
sns.lineplot(data=data, x='Period', y='Net Income', label='Net Income', marker='o', ax=ax)
ax.set_title("Revenue and Net Income Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Amount")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Assets vs. Liabilities Over Time
st.subheader("Assets vs. Liabilities Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=data, x='Period', y='Assets', label='Assets', marker='o', color='green', ax=ax)
sns.lineplot(data=data, x='Period', y='Liabilities', label='Liabilities', marker='o', color='red', ax=ax)
ax.set_title("Assets vs. Liabilities Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Amount")
ax.legend()
ax.grid(True)
st.pyplot(fig)
data['ROA (%)'] = pd.to_numeric(data['ROA (%)'].str.replace('.', '', regex=True), errors='coerce')
data['ROE (%)'] = pd.to_numeric(data['ROE (%)'].str.replace('.', '', regex=True), errors='coerce')
data['Debt to Equity'] = pd.to_numeric(data['Debt to Equity'].str.replace('.', '', regex=True), errors='coerce')
data['percentage  Debt to Equity'] = pd.to_numeric(data['percentage  Debt to Equity'].str.replace(',', '.').str.replace('%', ''), errors='coerce')
# Convert 'Period' to datetime
data['Period'] = pd.to_datetime(data['Period'], format='%m/%d/%Y')

# Page Header
st.title("Seasonal Decomposition of Revenue Data")

# Sidebar: Select Company
companies = data['Company'].unique()
selected_company = st.sidebar.selectbox("Select a Company:", companies)

# Filter data for the selected company
company_data = data[data['Company'] == selected_company]

# Display message if no data is available for the selected company
if company_data.empty:
    st.warning(f"No data available for {selected_company}.")
else:
    # Set 'Period' as the index and sort the data
    company_data = company_data.set_index('Period').sort_index()

    # Check if there are enough observations (at least 24 for period=12)
    if len(company_data) >= 24:
        st.subheader(f"Seasonal Decomposition for {selected_company}")
        revenue_series = company_data['Revenue']

        # Seasonal decomposition (assuming monthly data, period=12)
        decomposition = seasonal_decompose(revenue_series, model='additive', period=12)

        # Plot decomposition results
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=axes[0], title='Observed', legend=False)
        decomposition.trend.plot(ax=axes[1], title='Trend', legend=False)
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal', legend=False)
        decomposition.resid.plot(ax=axes[3], title='Residuals', legend=False)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning(f"Not enough data points for seasonal decomposition for {selected_company}. At least 24 observations are required.")
data['Year'] = data['Period'].dt.year
data['Quarter'] = data['Period'].dt.quarter
data['Revenue Growth'] = data.groupby('Company')['Revenue'].pct_change()

# Page Header
st.title("Revenue Prediction for Selected Company")

# Sidebar: Select Company
companies = data['Company'].unique()
selected_company = st.sidebar.selectbox("Select a Company for Analysis:", companies)

# Filter the data for the selected company
company_data = data[data['Company'] == selected_company]

if company_data.empty:
    st.warning(f"No data available for {selected_company}.")
else:
    # Set 'Period' as index and drop it from features
    company_data = company_data.set_index('Period')
    X = company_data.drop(columns=['Revenue'])  # Features
    y = company_data['Revenue']  # Target

    # Identify numeric and categorical columns
    numeric_columns = X.select_dtypes(include=['number']).columns
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Create preprocessors
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Build the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display Metrics
    st.subheader(f"Performance Metrics for {selected_company}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R-Squared (R²):** {r2:.2f}")

    # Show Predictions
    st.subheader("Predicted vs Actual Revenue")
    results = pd.DataFrame({
        "Actual Revenue": y_test,
        "Predicted Revenue": y_pred
    }).reset_index(drop=True)
    st.write(results)

    # Plot Predicted vs Actual Revenue
    st.line_chart(results)
# Ensure 'Period' column is correctly parsed as datetime
data['Period'] = pd.to_datetime(data['Period'], format='%m/%d/%Y')

# Filter Siemens data and set 'Period' as the index
company_data = data[data['Company'] == 'Siemens AG']
company_data = company_data.set_index('Period').sort_index()
st.title("Revenue Forecasting for Selected Company using ARIMA and SARIMA")

# Sidebar: Select Company
companies = data['Company'].unique()
selected_company = st.sidebar.selectbox("Select a Company for Forecasting:", companies)

# Filter the data for the selected company
company_data = data[data['Company'] == selected_company]

if company_data.empty:
    st.warning(f"No data available for {selected_company}.")
else:
    # Fill missing values if needed (forward fill for simplicity)
    company_data['Revenue'] = company_data['Revenue'].fillna(method='ffill')

    # Plot the Revenue to inspect
    st.subheader(f"{selected_company} Revenue Over Time")
    plt.figure(figsize=(10, 6))
    plt.plot(company_data['Period'], company_data['Revenue'])
    plt.title(f'{selected_company} Revenue Over Time')
    plt.xlabel('Period')
    plt.ylabel('Revenue')
    st.pyplot(plt)

    # Split the data into training and testing datasets
    train_size = int(len(company_data) * 0.8)
    train, test = company_data['Revenue'][:train_size], company_data['Revenue'][train_size:]

    # --- ARIMA Model ---
    arima_model = ARIMA(train, order=(5, 1, 0))  # AR(5), I(1), MA(0)
    arima_model_fit = arima_model.fit()

    # Forecasting using ARIMA
    arima_forecast = arima_model_fit.forecast(steps=len(test))

    # --- SARIMA Model ---
    sarima_model = SARIMAX(train, order=(5, 1, 0), seasonal_order=(1, 1, 1, 12))  # Example seasonal parameters (P, D, Q, S)
    sarima_model_fit = sarima_model.fit()

    # Forecasting using SARIMA
    sarima_forecast = sarima_model_fit.forecast(steps=len(test))

    # Plot the results for comparison
    st.subheader(f"Revenue Forecasting Comparison for {selected_company}")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Actual vs ARIMA Forecast
    axes[0].plot(test.index, test, color='blue', label='Actual Revenue')
    axes[0].plot(test.index, arima_forecast, color='red', label='ARIMA Forecasted Revenue')
    axes[0].set_title(f'{selected_company} Revenue Forecasting - ARIMA')
    axes[0].legend()

    # Actual vs SARIMA Forecast
    axes[1].plot(test.index, test, color='blue', label='Actual Revenue')
    axes[1].plot(test.index, sarima_forecast, color='green', label='SARIMA Forecasted Revenue')
    axes[1].set_title(f'{selected_company} Revenue Forecasting - SARIMA')
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # --- Evaluation ---
    # Evaluate the ARIMA model
    arima_mse = mean_squared_error(test, arima_forecast)
    arima_rmse = arima_mse ** 0.5
    st.write(f"**ARIMA RMSE:** {arima_rmse:.2f}")

    # Evaluate the SARIMA model
    sarima_mse = mean_squared_error(test, sarima_forecast)
    sarima_rmse = sarima_mse ** 0.5
    st.write(f"**SARIMA RMSE:** {sarima_rmse:.2f}")