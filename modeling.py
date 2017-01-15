import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_NN():
    X = np.load('./data/X.npy')
    y = np.load('./data/y.npy')
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    clf = MLPClassifier(activation = 'relu', alpha=1e-5, hidden_layer_sizes = (1000, 1000, 500, 500, 100, 100), random_state = 1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(accuracy_score(y_test, pred))
    # print(le.inverse_transform(y_test))
    # print(le.inverse_transform(pred))


if __name__ == '__main__':
    train_NN()