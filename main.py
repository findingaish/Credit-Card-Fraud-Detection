from src.data_processing import load_data, preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.predict import predict
import joblib

def main():
    # Step 1: Load and preprocess data
    train_file = 'data/fraudTrain.csv'
    test_file = 'data/fraudTest.csv'
    train_data, test_data = load_data(train_file, test_file)
    X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
    
    # Step 2: Train model
    model = train_model(X_train, y_train)
    
    # Step 3: Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Step 4: Save model
    save_model(model, 'models/saved_model.pkl')
    
    # Step 5: Example prediction
    predict(model)

def save_model(model, filename):
    """ Save the trained model """
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    main()
