# preprocessing.py

import pandas as pd

# 데이터 로드 함수
def load_data(filepath):
    """Load the data from a CSV file."""
    data = pd.read_csv(filepath)
    print('데이터 로드 완료')
    return data

# 결측치 제거 함수
def remove_missing_values(data):
    """Remove rows with missing values."""
    data = data.dropna()
    print('결측치 제거 완료')
    return data

# 필요없는 칼럼 제거 함수
def drop_columns(data):
    data = data.drop(['id'], axis=1)
    print('필요없는 칼럼 제거 완료')
    return data

# 범주형 변수와 연속형 변수 구분 함수
def get_numerical_categorical_columns():
    """Get lists of numerical and categorical columns."""
    numerical = ['Age', 'Annual_Premium', 'Vintage']
    categorical = ['Gender', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel']
    categorical_label = ['Region_Code', 'Vehicle_Age', 'Policy_Sales_Channel']
    categorical_onehot = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Damage']
    return numerical, categorical, categorical_label, categorical_onehot

# 범주형 변수 더미화 함수
def dummy_encode_categorical_columns(data, categorical_columns):
    """Dummy encode the categorical columns."""
    data = pd.get_dummies(data, columns=categorical_columns)
    print('범주형 변수 더미화 완료')
    return data

# 범주형 변수 Label Encoding 함수
def label_encode_categorical_columns(data, categorical_columns):
    """Label encode the categorical columns."""
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    print('범주형 변수 Label Encoding 완료')
    return data, label_encoders

# 데이터 저장 함수
def save_data(data, filepath):
    """Save the preprocessed data to a CSV file."""
    data.to_csv(filepath, index=False)
    print('데이터 저장 완료')

# pickle 파일로 저장 함수
def save_label_encoders(label_encoders, filepath):
    """Save the label encoders to a pickle file."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(label_encoders, f)
    print('Label Encoders 저장 완료')

###############################
# 전처리 함수
###############################
def preprocess_data(input_filepath, output_filepath):
    """Complete preprocessing pipeline."""
    data = load_data(input_filepath)
    data = remove_missing_values(data)
    data = drop_columns(data)
    save_data(data, output_filepath)

###############################
# 더미화 함수
###############################
def dummy_encoding(input_filepath, output_filepath):
    data = pd.read_csv(input_filepath)
    numerical_columns, categorical_columns, categorical_labels, categorical_onehots = get_numerical_categorical_columns()
    data = dummy_encode_categorical_columns(data, categorical_onehots)
    data, label_encoders = label_encode_categorical_columns(data, categorical_labels)
    save_data(data, output_filepath)
    save_label_encoders(label_encoders, '../models/label_encoders.pkl')




# 실행 예제
if __name__ == "__main__":
    # 데이터 전처리 함수 실행
    preprocess_data('../data/test.csv', '../data/test_preprocessed.csv')
    # 범주형 변수 더미화 함수 실행
    dummy_encoding('../data/test_preprocessed.csv', '../data/test_dummy_encoded.csv')