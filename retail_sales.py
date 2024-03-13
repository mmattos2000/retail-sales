%matplotlib inline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot

# Load the dataset
file_path = '/Users/melissamattos/Downloads/retail_sales_dataset.csv'  # Update this to the path of your dataset
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sales Analysis

# 1. Overall Sales Trend (Monthly Sales)
monthly_sales = data.groupby(pd.Grouper(key='Date', freq='M')).sum()['Total_Amount']

# 2. Sales by Product Category
sales_by_category = data.groupby('Product_Category').sum()['Total_Amount'].sort_values(ascending=False)

# 3. Purchases by Gender
purchases_by_gender = data.groupby('Gender').sum()['Total_Amount']

# 4. Age Distribution of Customers
age_distribution = data['Age'].value_counts().sort_index()

# Visualizations
fig, axs = plt.subplots(4, 1, figsize=(10, 20))

# Monthly Sales Trend
axs[0].plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-', color='b')
axs[0].set_title('Monthly Sales Trend')
axs[0].set_xlabel('Month')
axs[0].set_ylabel('Total Sales')

# Sales by Product Category
axs[1].bar(sales_by_category.index, sales_by_category.values, color='orange')
axs[1].set_title('Sales by Product Category')
axs[1].set_xlabel('Product Category')
axs[1].set_ylabel('Total Sales')

# Purchases by Gender
axs[2].bar(purchases_by_gender.index, purchases_by_gender.values, color='green')
axs[2].set_title('Purchases by Gender')
axs[2].set_xlabel('Gender')
axs[2].set_ylabel('Total Sales')

# Age Distribution of Customers
axs[3].bar(age_distribution.index, age_distribution.values, color='purple')
axs[3].set_title('Age Distribution of Customers')
axs[3].set_xlabel('Age')
axs[3].set_ylabel('Number of Customers')

plt.tight_layout()
plt.show()


# Sales Forecasting

# Display autocorrelation to help determine ARIMA parameters
autocorrelation_plot(monthly_sales)
plt.show()

# Note: Adjust the ARIMA model parameters (p,d,q) as needed based on autocorrelation plot and AIC.
p = 1  # AR term
d = 1  # Differencing order
q = 1  # MA term

# Fit the model
model = ARIMA(monthly_sales, order=(p, d, q))
model_fit = model.fit()

# Forecast the next 12 months
forecast_dates = model_fit.forecast(steps=12)

# Visualize the forecast
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales, label='Historical Monthly Sales')
plt.plot(pd.date_range(monthly_sales.index[-1], periods=13, closed='right'), forecast, label='Forecasted Monthly Sales', linestyle='--')
plt.title('Sales Forecast for the Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt


# For Age vs. Total_Amount
plt.figure(figsize=(10, 6))
sns.regplot(x='Age', y='Total_Amount', data=data, ci=None)
plt.title('Age vs. Total Amount')
plt.show()

# For Quantity vs. Total_Amount
plt.figure(figsize=(10, 6))
sns.regplot(x='Quantity', y='Total_Amount', data=data, ci=None)
plt.title('Quantity vs. Total Amount')
plt.show()

# For Price_per_Unit vs. Total_Amount
plt.figure(figsize=(10, 6))
sns.regplot(x='Price_per_Unit', y='Total_Amount', data=data, ci=None)
plt.title('Price per Unit vs. Total Amount')
plt.show()

correlation_age = data['Age'].corr(data['Total_Amount'])
correlation_quantity = data['Quantity'].corr(data['Total_Amount'])
correlation_price_per_unit = data['Price_per_Unit'].corr(data['Total_Amount'])

print(f"Correlation (Age vs. Total Amount): {correlation_age}")
print(f"Correlation (Quantity vs. Total Amount): {correlation_quantity}")
print(f"Correlation (Price per Unit vs. Total Amount): {correlation_price_per_unit}")
