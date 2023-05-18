from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class KnnModel:
    def __init__(self, n_neighbors):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def train_model(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features for KNN
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        self.model.fit(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        print(f'Validation accuracy: {val_acc}')

    def test_model(self, X, y):
        X_test = self.scaler.transform(X)
        y_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y, y_pred)
        print(f'Test accuracy: {test_acc}')
        return y_pred
