import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_classifier_and_predict():
    X = np.load('./data/X.npy')
    y = np.load('./data/labels.npy')
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # clf = MLPClassifier(activation = 'relu', alpha=1e-5, hidden_layer_sizes = (30,30), random_state = 1)
    # clf.fit(X_train, y_train)
    # pred = clf.predict(X_test)

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    print(accuracy_score(y_test, pred))
    # print(le.inverse_transform(y_test))
    # print(le.inverse_transform(pred))


if __name__ == '__main__':
    train_classifier_and_predict()