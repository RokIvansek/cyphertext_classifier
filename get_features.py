import numpy as np

def import_data():
    cyphertexts = np.load('./data/cyphertexts.npy')
    labels = np.load('./data/labels.npy')
    return cyphertexts, labels

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def encode_letters_as_features(cyphertexts, labels):
    X = [list(element) for element in cyphertexts]
    X = np.array(X)
    # Surely there is a better way...
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = alphabet.index(X[i, j])
    X = np.array(X, dtype=float)
    y = np.array(labels)
    return X, y

def count_letters(cpyhertexts):
    # X = np.zeros((cyphertexts.shape[0], len(alphabet)))
    return np.array([[c.count(letter) for letter in alphabet] for c in cyphertexts])

def count_adjacent_duplicates(cyphertexts):
    counts = []
    for c in cyphertexts:
        counter = 0
        for i in range(len(c)-1):
            if c[i] == c[i+1]: counter += 1
        counts.append(counter)
    return np.array(counts).reshape((len(cyphertexts), 1))

def combine_and_export():
    X = np.hstack((count_letters(cyphertexts),
                   count_adjacent_duplicates(cyphertexts)))
    np.save('./data/X', X)

if __name__ == '__main__':
    cyphertexts, labels = import_data()
    combine_and_export()
    # print(cyphertexts.shape, labels.shape)
    # print(cyphertexts[0], labels[0])