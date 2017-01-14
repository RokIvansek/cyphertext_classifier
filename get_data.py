import sys
import random
from pycipher import Affine, Vigenere

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def read_text(filepath, n):
    file = open(filepath, encoding="Latin-1")
    texts = []
    i = 0
    for row in file:
        _, _, text = row.partition('\t')
        text = text.replace(' ', '')
        # print(text)
        # print(len(text))
        if len(text) > n:
            text = text[:n]
            texts.append(text)
            i += 1
        if i == 1000:
            break
    file.close()
    return texts

def encrypt(plaintexts):
    # For now use only Affine, Vigenere
    cyphertexts = []
    labels = []
    # Affine
    bs = list(range(26))
    aas = [1,3,5,7,9,11,15,17,19,21,23,25]
    # Vigenere
    for x in plaintexts:
        # Affine
        affine = Affine(a = random.choice(aas), b = random.choice(bs)) # Encrypt with random keys
        y = affine.encipher(x)
        cyphertexts.append(y)
        labels.append("affine")
    for x in plaintexts:
        # Vigenere
        l = random.choice(list(range(1,20)))
        keyVigenere = [random.choice(alphabet) for _ in range(l)]
        keyVigenere = ''.join(keyVigenere)
        vigenere = Vigenere(keyVigenere)
        y = vigenere.encipher(x)
        cyphertexts.append(y)
        labels.append("vigenere")
    return cyphertexts, labels

def export(cyphertexts, labels):
    # TODO: Export chypherthexts and labels to a txt file. One string per line. First write label than the cypertext string.
    return

if __name__ == '__main__':
    texts = read_text(sys.argv[1], 1000)
    print("number of plaintexts:", len(texts))
    cyphertexts, labels = encrypt(texts)
    print(len(cyphertexts))
    print(len(labels))
