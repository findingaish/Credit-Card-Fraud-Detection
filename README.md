# Credit-Card-Fraud-Detection

This repository contains a fraud detection project focusing on analyzing credit card transactions to identify fraudulent activities using machine learning algorithms.

## Dataset

The dataset used in this project includes simulated credit card transactions:
- `fraudTrain.csv`: Training dataset containing legitimate and fraud transactions.
- `fraudTest.csv`: Test dataset for evaluating the trained models.

### Dataset Source
The dataset is sourced from [https://drive.google.com/drive/folders/1sDzIPjCmNZ9lWaXfcAqIIx4NZchYG4OP].

### Dataset Description
- `trans_date_trans_time`: Date and time of the transaction.
- `cc_num`: Credit card number.
- `merchant`: Name of the merchant.
- `category`: Category of the transaction.
- `amt`: Transaction amount.
- `first`, `last`: First and last name of the cardholder.
- `gender`: Gender of the cardholder.
- `street`, `city`, `state`, `zip`: Address information.
- `lat`, `long`: Latitude and longitude of the transaction location.
- `city_pop`: Population of the city.
- `job`: Occupation of the cardholder.
- `dob`: Date of birth of the cardholder.
- `trans_num`: Transaction number.
- `unix_time`: Unix timestamp of the transaction.
- `merch_lat`, `merch_long`: Latitude and longitude of the merchant location.
- `is_fraud`: Binary indicator (1 for fraudulent, 0 for legitimate transaction).

### Dependencies
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, joblib (for model serialization)

### Model Training
- Use model_training.py to train machine learning models on the dataset.
python src/model_training.py

### Model Evaluation
- Use model_evaluation.py to evaluate model performance on the test dataset.
  python src/model_evaluation.py

### Prediction
- Use predict.py to make predictions on new data using the trained model.
  python src/predict.py


