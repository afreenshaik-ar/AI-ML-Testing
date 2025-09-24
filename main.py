import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_and_preprocess_data():
    import pandas as pd
    base_dir = os.path.dirname(os.path.abspath(__file__))  # always points to main.py folder
    file_path = os.path.join(base_dir, "data", "titanic.csv")  # build safe path
    data = pd.read_csv(file_path)
    # Handle missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

    # Drop unnecessary columns
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    return data



def train_model(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Split features and target
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model, accuracy



if __name__ == "__main__":
    data = load_and_preprocess_data()
    train_model(data)
