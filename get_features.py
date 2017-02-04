import numpy as np
from suffix_trees import STree

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

def count_letters(cyphertexts, export=False):
    # X = np.zeros((cyphertexts.shape[0], len(alphabet)))
    X = np.array([[c.count(letter) for letter in alphabet] for c in cyphertexts])
    if export:
        np.save('./data/dist_of_letters', X)
    return X

def count_adjacent_duplicates(cyphertexts, export=False):
    counts = []
    for c in cyphertexts:
        counter = 0
        for i in range(len(c)-1):
            if c[i] == c[i+1]: counter += 1
        counts.append(counter)
    X = np.array(counts).reshape((len(cyphertexts), 1))
    if export:
        np.save('./data/no_adj_dups', X)
    return X

# TODO: Find a fast way to do this for arbitrary length of substring.
# Counts the digrams that repeat at least one and gives the no of repeats for the most freaquent digram
def count_repeating_bigrams(cyphertexts, export=False):
    counts = []
    maxis = []
    for c in cyphertexts:
        count = 0
        maxi = 0
        for i in range(len(alphabet)):
            for j in range(len(alphabet)):
                seq = alphabet[i] + alphabet[j]
                n = c.count(seq)
                if n > 0:
                    count += 1
                if n > maxi:
                    maxi = n
        counts.append(count)
        maxis.append(maxi)
    X = np.column_stack((counts, maxis))
    if export:
        np.save('./data/digrams', np.array(X))
    return X

def index_of_coincidence(cyphertexts, export=False):
    iocs = []
    for c in cyphertexts:
        l = len(c)
        freqs = np.array([c.count(letter) for letter in alphabet])
        row = [np.sum([(f*(f-1))/(l*(l-1)) for f in freqs])]
        iocs.append(row)
    X = np.array(iocs)
    if export:
        np.save('./data/iocs', X)
    return X

def combine_and_export():
    X = np.hstack((count_letters(cyphertexts),
                   count_adjacent_duplicates(cyphertexts),
                   index_of_coincidence(cyphertexts),
                   count_repeating_bigrams(cyphertexts)))
    np.save('./data/X', X)

if __name__ == '__main__':
    cyphertexts, labels = import_data()
    combine_and_export()
    # print(count_repeating_digrams(cyphertexts))
    # iocs = index_of_coincidence(cyphertexts, (4,10))
    # print(iocs.shape)
    # print(iocs)
    # print(cyphertexts.shape, labels.shape)
    # print(cyphertexts[0], labels[0])