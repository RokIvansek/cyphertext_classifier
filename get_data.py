import sys
import random
import numpy as np
import pycipher
import re

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
    affine = pycipher.Affine(a=random.choice(aas), b=random.choice(bs))  # Encrypt with random keys
    y = affine.encipher(x)
    return y

def e_vigenere(x):
    l = random.choice(list(range(1, 20))) #hardcoded that the key length is between 1 and 20 char long
    key_vigenere = [random.choice(alphabet) for _ in range(l)]
    key_vigenere = ''.join(key_vigenere)
    vigenere = pycipher.Vigenere(key_vigenere)
    y = vigenere.encipher(x)
    return y

def e_ADFGVX(x):
    # TODO: Fix so that key and keyword will be random.
    adfgvx = pycipher.ADFGVX(key='PH0QG64MEA1YL2NOFDXKR3CVS5ZW7BJ9UTI8', keyword='GERMAN')
    y = adfgvx.encipher(x)
    return y

def e_ceasar(x):
    key_c = random.choice(range(1,25))
    ceasar = pycipher.Caesar(key=key_c)
    y = ceasar.encipher(x)
    return y

def e_permutation(x):
    key_p = ''.join(random.sample(alphabet,len(alphabet)))
    return pycipher.SimpleSubstitution(key=key_p).encipher(x)

def encrypt(plaintexts):
    cyphertexts = []
    labels = []
    encryptions = [e_affine, e_ceasar, e_ADFGVX, e_vigenere, e_permutation]
    for x in plaintexts:
        for encryption in encryptions:
            cyphertexts.append(encryption(x))
            labels.append(re.search('function e_(.*) at', str(encryption)).group(1))
        # # Affine
        # y = e_affine(x)
        # cyphertexts.append(y)
        # labels.append("affine")
        # # # Vigenere
        # y = e_vigenere(x)
        # cyphertexts.append(y)
        # labels.append("vigenere")
        # # # ADFGVX
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
    for i in range(len(np.unique(labels))):
        print("Example of", labels[i], "cypher.")
        print(cyphertexts[i])
    export(cyphertexts, labels)
    # X, y = prepare_data(cyphertexts, labels)
    # print(X)
    # print(y)
    # export(X, y)
