import streamlit as st
import pandas as pd
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns

# Initialize and load the XGBoost model
model = xgb.XGBRegressor()
model.load_model('xgb_revenue_predictor.model')

st.title('Revenue Prediction tool for Kenyan County Governments')

# File upload logic
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset")
    st.write(data.head())

    # EDA and Visualization
    if st.checkbox('Show EDA and Visualizations'):
        st.subheader("Basic EDA")

        # Basic stats
        st.write("Descriptive Statistics")
        st.write(data.describe())

        # Check for missing values
        st.write("Missing values in dataset:")
        missing_data = data.isnull().sum()
        st.write(missing_data[missing_data > 0])

        # Data visualizations
        st.subheader("Data Visualizations")
        
        # Distribution of numeric features
        numeric_columns = data.select_dtypes(['float', 'int']).columns.tolist()
        select_column = st.selectbox('Select Column to Display Distribution', numeric_columns)
        if st.button('Show Distribution'):
            fig, ax = plt.subplots()
            sns.histplot(data[select_column], kde=True, ax=ax)
            st.pyplot(fig)
            
          
                  # Time Series plot (assuming a date column exists)
        if 'Date' in data.columns:
            st.subheader("Time Series Analysis")
            data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' column is in datetime format

            # Option to select a numeric column for time series plotting
            time_series_column = st.selectbox('Select Time Series Column', numeric_columns)
            if st.button('Show Time Series Plot'):
                fig, ax = plt.subplots()
                data.set_index('Date')[time_series_column].plot(ax=ax)
                ax.set_title(f'Time Series Plot of {time_series_column}')
                ax.set_ylabel(time_series_column)
                ax.grid(True)
                st.pyplot(fig)
            
        # Time Series plot (assuming a date column exists)
        if 'Date' in data.columns:
            st.subheader("Time Series Analysis")
            data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' column is in datetime format

            # Calculating the 12-month moving average
            if 'Total Daily Amount' in data.columns:
                data['12M_Moving_Average'] = data['Total Daily Amount'].rolling(window=12).mean()

            # Option to select a numeric column for time series plotting
            time_series_column = st.selectbox('Select Time Series Column', numeric_columns)
            if st.button('Show Time Series Plot'):
                fig, ax = plt.subplots()
                data.set_index('Date')[time_series_column].plot(ax=ax)
                ax.set_title(f'Time Series Plot of {time_series_column}')
                ax.set_ylabel(time_series_column)
                ax.grid(True)
                st.pyplot(fig)
            
            # Plotting Trend of Total Daily Amount Over Time
            if st.button('Show Trend of Total Daily Amount and Moving Average'):
                plt.figure(figsize=(12, 6))
                plt.plot(data['Date'], data['Total Daily Amount'], label='Total Daily Amount', color='blue')
                plt.plot(data['Date'], data['12M_Moving_Average'], label='12-Month Moving Average', color='red', linestyle='--')
                plt.title('Total Daily Amount with 12-Month Moving Average')
                plt.xlabel('Date')
                plt.ylabel('Total Daily Amount')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot()  # Display the plot in Streamlit


        
        # # Correlation heatmap
        # if numeric_columns:
        #     if st.button('Show Correlation Heatmap'):
        #         st.subheader("Correlation Heatmap")
        # plt.figure(figsize=(10, 7))
        # sns.heatmap(data[numeric_columns].corr(), annot=True, fmt=".2f", cmap='coolwarm')  # Added a color map for better visualization
        # st.pyplot()
        # plt.clf()  # Clear the figure to prevent overlap


        # # Pair plot (scatter plots and histograms)
        # if st.button("Show Pair Plot"):
        #     st.subheader("Pair Plot")
        #     pair_plot_data = data[numeric_columns].sample(frac=0.1)  # Reduce data size by sampling
        #     sns.pairplot(pair_plot_data)
        #     st.pyplot()
        #     plt.clf()  # Clear the figure after plotting


        # # Box plots for each numeric feature
        # if st.button("Show Box Plots"):
        #     st.subheader("Box Plots for Numeric Features")
        #     for column in numeric_columns:
        #         fig, ax = plt.subplots()
        # sns.boxplot(x=data[select_column])
        # st.pyplot()
        
        # plt.clf()  # Clear the figure after plotting


        # Categorical data analysis (if categorical columns exist)
        categorical_columns = data.select_dtypes(['object']).columns.tolist()
        if categorical_columns:
            st.subheader("Categorical Data Analysis")
            select_categorical = st.selectbox('Select Categorical Column', categorical_columns)
            if st.button('Show Value Counts'):
                fig, ax = plt.subplots()
                sns.countplot(x=select_categorical, data=data)
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        # Time Series plot (assuming a date column exists)
        if 'Date' in data.columns:
            st.subheader("Time Series Analysis")
            data['Date'] = pd.to_datetime(data['Date'])
            time_series_column = st.selectbox('Select Time Series Column', numeric_columns)
            if st.button('Show Time Series Plot'):
                fig, ax = plt.subplots()
                data.set_index('Date')[time_series_column].plot(ax=ax)
                ax.set_title(f'Time Series Plot of {time_series_column}')
                ax.set_ylabel(time_series_column)
                ax.grid(True)
                st.pyplot(fig)

# Prediction Interface
st.subheader('Predict Total Monthly Amount with XGBoost')

date_input = st.date_input("Select Date", datetime.now())
month1_amount = st.number_input('Enter amount for Month 1:', step=1.0, format="%.2f")
month2_amount = st.number_input('Enter amount for Month 2:', step=1.0, format="%.2f")
month3_amount = st.number_input('Enter amount for Month 3:', step=1.0, format="%.2f")

# Convert date_input to pandas Timestamp
date_input = pd.to_datetime(date_input)

# Prepare the input data for prediction
input_data = {
    'Year': date_input.year,
    'Month': date_input.month,
    'Day': date_input.day,
    'Dayofweek': date_input.dayofweek,
    'Dayofyear': date_input.dayofyear,
    'Is_month_start': int(date_input.is_month_start),
    'Is_month_end': int(date_input.is_month_end),
    'Is_quarter_start': int(date_input.is_quarter_start),
    'Is_quarter_end': int(date_input.is_quarter_end),
    'Is_year_start': int(date_input.is_year_start),
    'Is_year_end': int(date_input.is_year_end),
    'Month1_Amount': month1_amount,
    'Month2_Amount': month2_amount,
    'Month3_Amount': month3_amount
}

if st.button('Predict Total Amount'):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    prediction_in_kshs = prediction * 135  # Modify the exchange rate as necessary
    st.write(f"The predicted total monthly amount for {date_input.strftime('%Y-%m-%d')} is Kshs {prediction_in_kshs:.2f}")

# Instructions to run the app
# Save this script as app.py and execute `streamlit run app.py` in your terminal.
