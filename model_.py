import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

class LoanDefaultModel:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load(self, data_path):
        return pd.read_excel(data_path)

    def preprocess(self, df):
        categorical_cols = ['sub_grade', 'term', 'home_ownership', 'purpose', 'application_type', 'verification_status']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        df.fillna(df.median(), inplace=True)
        X = df.drop(columns=['customer_id', 'transaction_date', 'loan_status'])
        y = df['loan_status']
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def train(self, X_train, y_train, model_type='logistic'):
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError("Invalid model type")
        model.fit(X_train, y_train)
        self.models[model_type] = model

    def test(self, X_test, y_test, model_type):
        model = self.models.get(model_type)
        if not model:
            raise ValueError(f"Model {model_type} is not trained yet.")
        y_pred = model.predict(X_test)
        print(f"Model: {model_type}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    def save_model(self, model_type, file_path):
        model = self.models.get(model_type)
        if not model:
            raise ValueError(f"Model {model_type} is not trained yet.")
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
            
if __name__ == "__main__":
    model_pipeline = LoanDefaultModel()

    # Load and preprocess the data
    print("Loading and preprocessing data...")
    data = model_pipeline.load("train_data.xlsx")
    X, y = model_pipeline.preprocess(data)

    # Split the data
    print("Splitting the data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    print("Training models...")
    model_pipeline.train(X_train, y_train, model_type='logistic')
    model_pipeline.train(X_train, y_train, model_type='random_forest')

    # Test models
    print("Testing models...")
    model_pipeline.test(X_test, y_test, model_type='logistic')
    model_pipeline.test(X_test, y_test, model_type='random_forest')
