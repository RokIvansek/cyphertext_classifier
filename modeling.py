import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def train_classifier_and_predict():
    X = np.load('./data/X.npy')
    y = np.load('./data/labels.npy')
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # clf = MLPClassifier(activation = 'relu', alpha=1e-5, hidden_layer_sizes = (100,200,100), random_state = 1)
    # clf.fit(X_train, y_train)
    # pred = clf.predict(X_test)

    # for n in [5000]:
    #     rf = RandomForestClassifier(n_estimators=n, criterion='entropy', n_jobs=-1)
    #     scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    #     print(n, "trees accuracy score:", np.mean(scores))

    rf = RandomForestClassifier(n_estimators=2000, criterion='entropy', n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    print(accuracy_score(y_test, pred))
    print(classification_report(y_test, pred, target_names=le.inverse_transform(np.arange(len(np.unique(y))))))

    # for cipher in ['affine', 'ceasar', 'permutation']:
    #     indices = y_test == cipher
    #     predictions = pred[indices]
    #     misclasifided_as = [(a, predictions[]) for a in np.unique(pred)]

    print(confusion_matrix(y_test, pred))
    # print(le.inverse_transform(y_test))
    # print(le.inverse_transform(pred))


if __name__ == '__main__':
    train_classifier_and_predict()