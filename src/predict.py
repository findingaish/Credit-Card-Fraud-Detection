import pandas as pd
import joblib

def predict(model):
    """ Example of prediction """
    # Example of new data for prediction
    new_data = pd.DataFrame({
        'trans_date_trans_time': ['2023-01-01 12:34:56'],
        'cc_num': ['1234567812345678'],
        'merchant': ['Store X'],
        'category': ['Grocery'],
        'amt': [100.0],
        'first': ['John'],
        'last': ['Doe'],
        'gender': ['M'],
        'street': ['123 Main St'],
        'city': ['New York'],
        'state': ['NY'],
        'zip': ['10001'],
        'lat': [40.7128],
        'long': [-74.0060],
        'city_pop': [8000000],
        'job': ['Engineer'],
        'dob': ['1990-01-01'],
        'trans_num': ['T123456789'],
        'unix_time': [1672536896],
        'merch_lat': [40.7128],
        'merch_long': [-74.0060]
    })
    
    # Preprocess new data (using the same scaler as in data_processing)
    scaler = StandardScaler()  
    # Fit scaler on training data
    scaler.fit(X_train)  
    new_data_scaled = scaler.transform(new_data)
    
    # Load saved model
    loaded_model = joblib.load('models/saved_model.pkl')
    
    # Make prediction
    prediction = loaded_model.predict(new_data_scaled)
    
    print("\nPrediction:")
    print(prediction)
