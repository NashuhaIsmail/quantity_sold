import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df_less_100 = pd.read_csv("C:\\Users\\nesa.nashuha\\OneDrive - Habib Jewels Sdn Bhd\\MC Project\\data\\combined data_10 to100.csv")
df_more_100 = pd.read_csv("C:\\Users\\nesa.nashuha\\OneDrive - Habib Jewels Sdn Bhd\\MC Project\\data\\combined data_more 100.csv")
df_less_10 = pd.read_csv("C:\\Users\\nesa.nashuha\\OneDrive - Habib Jewels Sdn Bhd\\MC Project\\data\\Combined data_less10.csv")

# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    reg_model = LinearRegression().fit(X_train, y_train)
    return reg_model, X_test, y_test

# Function to display scatter plot
def display_scatter_plot(df, x_vars, y_var):
    sns.pairplot(df, x_vars=x_vars, y_vars=y_var, height=4, aspect=1, kind='scatter')
    st.pyplot()

# Function to display distribution plot
def display_distribution_plot(series):
    sns.distplot(series)
    st.pyplot()

# Streamlit UI
st.title("Gold Demand Prediction")

# Choose category
category_options = ["Casual Buyer: Less than 10 units per transaction", "Regular Buyer: Less than 100 units per transaction", "High-Volume Buyer: More than 100 units per transaction"]
category_option = st.radio("Select Category:", category_options)

# Display scatter plot based on category
if category_option == "Casual Buyer: Less than 10 units per transaction":
    display_scatter_plot(df_less_10, ['exchange rate my-usd', 'Gold Price 1g/selling'], 'Quantity Sold')
    display_distribution_plot(df_less_10['Quantity Sold'])
    X_category = df_less_10[['Gold Price 1g/selling', 'exchange rate my-usd']]
    y_category = df_less_10['Quantity Sold']
elif category_option == "Regular Buyer: Less than 100 units per transaction":
    display_scatter_plot(df_less_100, ['exchange rate my-usd', 'Gold Price 1g/selling'], 'Quantity Sold')
    display_distribution_plot(df_less_100['Quantity Sold'])
    X_category = df_less_100[['Gold Price 1g/selling', 'exchange rate my-usd']]
    y_category = df_less_100['Quantity Sold']
else:
    display_scatter_plot(df_more_100, ['exchange rate my-usd', 'Gold Price 1g/selling'], 'Quantity Sold')
    display_distribution_plot(df_more_100['Quantity Sold'])
    X_category = df_more_100[['Gold Price 1g/selling', 'exchange rate my-usd']]
    y_category = df_more_100['Quantity Sold']

# Input for target value
target_value = st.number_input("Enter the actual target value:", min_value=0)

# Train the model based on category
model, X_test, y_test = train_model(X_category, y_category)

# Make predictions
prediction = model.predict([[target_value, 0]])  # Assuming exchange rate my-usd is 0 for simplicity

# Display results
st.write("Actual Value:", target_value)
st.write("Predicted Value:", prediction[0])
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


