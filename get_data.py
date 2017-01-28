import sys
import random
import numpy as np
from pycipher import Affine, Vigenere, ADFGVX

# alphabet = 'abcdefghijklmnopqrstuvwxyz'
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def get_plaintexts(filepath, l, n):
    file = open(filepath, encoding="Latin-1")
    texts = []
    i = 0
    for row in file:
        _, _, text = row.partition('\t')
        text = text.replace(' ', '')
        # print(text)
        # print(len(text))
        if len(text) > l:
            text = text[:l]
            texts.append(text)
            i += 1
        if i == n:
            break
    file.close()
    return texts

def e_affine(x):
    bs = list(range(26))
    aas = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]
    affine = Affine(a=random.choice(aas), b=random.choice(bs))  # Encrypt with random keys
    y = affine.encipher(x)
    return y

def e_vigenere(x):
    l = random.choice(list(range(1, 20))) #hardcoded that the key length is between 1 and 20 char long
    keyVigenere = [random.choice(alphabet) for _ in range(l)]
    keyVigenere = ''.join(keyVigenere)
    vigenere = Vigenere(keyVigenere)
    y = vigenere.encipher(x)
    return y

def e_ADFGVX(x):
    adfgvx = ADFGVX(key='PH0QG64MEA1YL2NOFDXKR3CVS5ZW7BJ9UTI8', keyword='GERMAN')
    y = adfgvx.encipher(x)
    return y

def encrypt(plaintexts):
    # For now use only Affine, Vigenere
    cyphertexts = []
    labels = []
    # Vigenere
    for x in plaintexts:
        # Affine
        y = e_affine(x)
        cyphertexts.append(y)
        labels.append("affine")
        # # Vigenere
        y = e_vigenere(x)
        cyphertexts.append(y)
        labels.append("vigenere")
        # # ADFGVX
        # y = e_ADFGVX(x)
        # cyphertexts.append(y)
        # labels.append("ADFGVX")
    return cyphertexts, labels

# def prepare_data(cyphertexts, labels):
#     X = [list(element) for element in cyphertexts]
#     X = np.array(X)
#     # Surely there is a better way...
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             X[i, j] = alphabet.index(X[i, j])
#     X = np.array(X, dtype=float)
#     y = np.array(labels)
#     return X, y

def export(cyphertexts, labels):
    np.save('./data/cyphertexts', np.array(cyphertexts))
    np.save('./data/labels', np.array(labels))
    # np.save('./data/X', X)
    # np.save('./data/y', y)

if __name__ == '__main__':
    plaintexts = get_plaintexts(sys.argv[1], 500, 1000)
    print("number of plaintexts:", len(plaintexts))
    cyphertexts, labels = encrypt(plaintexts)
    print(len(cyphertexts))
    print(len(labels))
    print(cyphertexts[0], labels[0])
    print(cyphertexts[1], labels[1])
    export(cyphertexts, labels)
    # X, y = prepare_data(cyphertexts, labels)
    # print(X)
    # print(y)
    # export(X, y)
