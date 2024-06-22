import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_file, test_file):
    """ Load train and test datasets """
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """ Preprocess data """
    # Convert datetime to Unix timestamp
    train_data['unix_time'] = pd.to_datetime(train_data['trans_date_trans_time']).astype(int) / 10**9
    test_data['unix_time'] = pd.to_datetime(test_data['trans_date_trans_time']).astype(int) / 10**9
    
    # Drop non-numeric and irrelevant columns
    drop_columns = ['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'first', 'last', 
                    'gender', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num', 
                    'merch_lat', 'merch_long']
    
    X_train = train_data.drop(['is_fraud'] + drop_columns, axis=1)
    y_train = train_data['is_fraud']
    
    X_test = test_data.drop(['is_fraud'] + drop_columns, axis=1)
    y_test = test_data['is_fraud']
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
