import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os

# Function to load the trained models from the folder
def load_models():
    models = {}
    model_dir = "C:/Users/kagam/Documents/Final Project/Models"
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    for model_file in model_files:
        sku = int(model_file.split("_")[1].split(".")[0])
        model_path = os.path.join(model_dir, model_file)
        model = joblib.load(model_path)
        models[sku] = model
    return models

# Load the trained models
models = load_models()

# Function to get SKUs for each functionality
def get_skus_for_functionality():
    functionality_skus = {
        'bluetooth speakers': [8],
        'bluetooth tracker': [22],
        'digital pencils': [26, 18],
        'fitness trackers': [43, 9],
        'flash drives': [24, 20],
        'headphones': [2, 3],
        'mobile phone accessories': [1, 31, 39, 5, 40, 15, 29, 38, 30, 17, 4],
        'portable smartphone chargers': [36, 27, 34, 35, 16],
        'selfie sticks': [33, 6, 21, 7, 23, 25, 37],
        'smartphone stands': [13, 41, 14, 28, 44, 42],
        'streaming sticks': [19, 12, 32, 11],
        'vr headset': [10]
    }
    return functionality_skus

# Function to get SKUs for each color
def get_color_for_skus():
    sku_colors = {
        'black' : [ 1, 8, 19, 31, 13, 12, 41, 26, 40, 14, 28, 18, 32, 44, 9, 42, 11],
        'blue' : [ 2, 27, 6, 7, 38, 16, 17],
        'gold' : [43, 37],
        'green' : [33, 21, 23, 25],
        'grey' : [34, 29, 30],
        'none' : [24, 20],
        'pink' : [39],
        'purple' : [3],
        'red' : [36, 5, 15, 35, 4],
        'white' : [10, 22]
    }
    return sku_colors

# Function to get SKUs for each vendor
def get_vendor_for_skus():
    sku_vendors = {
        '1' : [2, 3],
        '2' : [22],
        '3' : [6, 7],
        '4' : [23],
        '5' : [41, 24, 40, 20],
        '6' : [ 1, 19, 31, 12, 26, 15, 29, 18, 44, 30, 17, 4],
        '7' : [37],
        '8' : [36, 33, 27, 21, 34, 35, 25, 16],
        '9' : [43, 10, 9],
        '10' : [ 8, 13, 39, 5, 14, 28, 38, 32, 42, 11]
    }
    return sku_vendors


# Main function to run the app
def main():
    # Input details in the left sidebar
    with st.sidebar:
        st.title('Select the functionality')
        #input variables
        week = st.date_input('Week')
        # Select functionality
        functionality_skus = get_skus_for_functionality()
        functionality = st.selectbox('Functionality', list(functionality_skus.keys()))
    
    # Input details in the main content
    # Input form for user input
    st.title('Unit Sales Prediction')
    
    # Select sku
    skus = functionality_skus[functionality]
    sku = st.selectbox('SKU', list(skus))
     
    # Get color for the sku
    sku_color = get_color_for_skus()
    def color_for_sku(sku):
        for color, skus in sku_color.items():
            if sku in skus:
                return color
        return None  # SKU not found in any color
    color = color_for_sku(sku)
    
    price = st.number_input('Price')
    # Input for previous two weeks' prices
    price_prev_week_1 = st.number_input('Price of SKU in Previous Week', key='price_prev_week_1')
    price_prev_week_2 = st.number_input('Price of SKU Two Weeks Ago', key='price_prev_week_2')
    
    # Get vendor for the sku
    sku_vendor = get_vendor_for_skus()
    def vendor_for_sku(sku):
        for vendor, skus in sku_vendor.items():
            if sku in skus:
                return vendor
        return None  # SKU not found in any vendor
    vendor = vendor_for_sku(sku)

    # Input prices for SKUs with the same functionality
    if len(skus) == 1:
        avg_price = price  # If only one SKU, average price is the same as the price of that SKU
    else:
        sku_prices = {}
        for sku_id in skus:
            sku_prices[sku_id] = st.number_input(f"Enter price for SKU {sku_id}")

        # Calculate average price
        avg_price = sum(sku_prices.values()) / len(sku_prices)

    # Use a checkbox for feat_main_page
    feat_main_page = st.checkbox('Feature on Main Page')

    price_difference = price - avg_price
    price_lag_difference_1 = price - price_prev_week_1
    price_lag_difference_2 = price - price_prev_week_2
    
    # Prepare data for prediction
    # Initializing StandardScaler
    scaler = StandardScaler()
    
    # Scale the price columns
    scaler = StandardScaler()
    scaled_price = scaler.fit_transform([[price]])[0][0]
    scaled_price_prev_week_1 = scaler.transform([[price_prev_week_1]])[0][0]
    scaled_price_prev_week_2 = scaler.transform([[price_prev_week_2]])[0][0]
    scaled_avg_price = scaler.transform([[avg_price]])[0][0]
    scaled_price_difference = scaler.transform([[price_difference]])[0][0]
    scaled_price_lag_difference_1 = scaler.transform([[price_lag_difference_1]])[0][0]
    scaled_price_lag_difference_2 = scaler.transform([[price_lag_difference_2]])[0][0]
    
    # Convert 'week' to Pandas datetime object
    week_date = pd.to_datetime(week)

    # Extract year, month, and week number
    year = week_date.year - 2016
    month = week_date.month
    week_number = week_date.isocalendar().week

    data = {
        'year': year,
        'month': month,
        'week_number': week_number,
        'price': scaled_price,
        'avg_price': scaled_avg_price,  # Add avg_price from the preprocessing step
        'price_difference': scaled_price_difference,  # Calculate price_difference
        'price_lag_1': scaled_price_prev_week_1,
        'price_lag_2': scaled_price_prev_week_2,
        'price_lag_difference_1': scaled_price_lag_difference_1,  # Calculate price_lag_difference_1
        'price_lag_difference_2': scaled_price_lag_difference_2,  # Calculate price_lag_difference_2
        'feat_main_page': 1 if feat_main_page else 0,  # Convert checkbox value to 1 or 0
        'vendor': vendor,
        'functionality_bluetooth tracker': 1 if functionality == 'bluetooth tracker' else 0,  # Create dummy variables for functionality
        'functionality_digital pencils': 1 if functionality == 'digital pencils' else 0,
        'functionality_fitness trackers': 1 if functionality == 'fitness trackers' else 0,
        'functionality_flash drives': 1 if functionality == 'flash drives' else 0,
        'functionality_headphones': 1 if functionality == 'headphones' else 0,
        'functionality_mobile phone accessories': 1 if functionality == 'mobile phone accessories' else 0,
        'functionality_portable smartphone chargers': 1 if functionality == 'portable smartphone chargers' else 0,
        'functionality_selfie sticks': 1 if functionality == 'selfie sticks' else 0,
        'functionality_smartphone stands': 1 if functionality == 'smartphone stands' else 0,
        'functionality_streaming sticks': 1 if functionality == 'streaming sticks' else 0,
        'functionality_vr headset': 1 if functionality == 'vr headset' else 0,
        'color_blue': 1 if color == 'blue' else 0,
        'color_gold': 1 if color == 'gold' else 0,
        'color_green': 1 if color == 'green' else 0,  # Create dummy variables for color
        'color_grey': 1 if color == 'grey' else 0,
        'color_none': 1 if color == 'none' else 0,
        'color_pink': 1 if color == 'pink' else 0,
        'color_purple': 1 if color == 'purple' else 0,
        'color_red': 1 if color == 'red' else 0,
        'color_white': 1 if color == 'white' else 0,
    }

    df_input = pd.DataFrame(data, index=[0])

    if st.button('Predict'):
        # Make prediction using the appropriate model for the selected SKU
        model = models.get(sku)
        if model is not None:
            prediction = model.predict(df_input)
            # Display prediction
            st.write('<h4>Predicted Sales:</h4>', unsafe_allow_html=True)
            st.write(f'<h4>{round(prediction[0])} Units</h4>', unsafe_allow_html=True)
        else:
            st.write("Model for the selected SKU is not available.")
if __name__ == "__main__":
    main()
