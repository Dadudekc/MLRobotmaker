# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import featuretools as ft
import asyncio

# Assuming shared_data_store.py is in the same directory or the module path is correctly set
from Utilities.shared_data_store import SharedDataStore, ModelRetrainer, ModelNotifier
from lightgbm import LGBMRegressor

def feature_importance_analysis(X_train, y_train):
    model = LGBMRegressor().fit(X_train, y_train)
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index = X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    return feature_importances

# Automated Feature Engineering Function
def automated_feature_engineering(data):
    es = ft.EntitySet(id='data')
    es.entity_from_dataframe(entity_id='df', dataframe=data, index='index')
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='df', max_depth=2)
    return feature_matrix

# Exploration Features Function
def explore_features(data):
    selected_features = [col for col in data.columns if col != 'target']
    for feature in selected_features:
        plt.figure(figsize=(10, 4))
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f'Distribution of {feature}')
        plt.show()
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = data[selected_features + ['target']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.show()

# Load Data Function
def load_data(filepath):
    return pd.read_csv(filepath)

# Clean Data Function
def clean_data(data):
    for col in data.columns:
        data[col].fillna(data[col].median(), inplace=True)
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
    return data

# Generate Features Function
def generate_features(data):
    data['lag1'] = data['value'].shift(1)
    data['rolling_mean3'] = data['value'].rolling(window=3).mean()
    data.fillna(0, inplace=True)
    return data

# Split Data Function
def split_data(data, test_size=0.2, random_state=None):
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Feature Selection Function
def feature_selection_lgbm(data, target_column, n_features=20):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    model = LGBMRegressor().fit(X, y)
    importances = model.feature_importances_
    indices = importances.argsort()[-n_features:]
    selected_features = X.columns[indices].tolist() + [target_column]
    return data[selected_features]

# Preprocess Data Function
def preprocess_data(data_path, n_features=20, random_state=42):
    data = load_data(data_path)
    data = clean_data(data)
    data = generate_features(data)
    data = automated_feature_engineering(data)
    data = feature_selection_lgbm(data, 'target', n_features)
    explore_features(data)
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test

def plot_feature_importances(feature_importances):
    # Take the top 20 features for visualization
    top_features = feature_importances.head(20)
    sns.barplot(x=top_features['importance'], y=top_features.index)
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

# Main async function to utilize SharedDataStore and observers
    
async def main():
    shared_data_store = SharedDataStore()
    model_retrainer = ModelRetrainer()
    model_notifier = ModelNotifier()

    # Register observers
    shared_data_store.register_observer(model_retrainer, interest="dataset_update")
    shared_data_store.register_observer(model_notifier, interest="model_update")


    # This block replaces the original call to preprocess_data in main
    data_path = shared_data_store.get_configuration('data_path', "path/to/your/dataset.csv")
    n_features = shared_data_store.get_configuration('n_features', 20)
    random_state = shared_data_store.get_configuration('random_state', 42)
    X_train, X_test, y_train, y_test = preprocess_data(data_path, n_features, random_state)
    shared_data_store.add_dataset("my_dataset", {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test})

if __name__ == "__main__":
    asyncio.run(main())
