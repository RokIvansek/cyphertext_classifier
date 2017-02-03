import sys
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from get_features import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def dim_reduction(X):
    ss = StandardScaler()
    pca = PCA(n_components=2)
    X = pca.fit_transform(ss.fit_transform(X))
    return X

def plot_scatter_2d(data, labels, subsample=1):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    _, X_test, _, y_test = train_test_split(X, labels, test_size=subsample, random_state=42)
    plt.scatter(X_test[:,0], X_test[:,1], s=10, c=y_test, alpha=0.5)
    plt.show()

def draw_distributions(dist_of_letters, labels):
    for cypher in np.unique(labels):
        indices = labels == cypher
        cypher_data = dist_of_letters[indices, :]
        x = np.mean(cypher_data, axis=0).astype(int)
        objects = (list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        y_pos = np.arange(len(objects))
        plt.ylim((0,300))
        plt.bar(y_pos, x, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Count')
        plt.title(cypher)
        plt.grid()
        plt.show()

def plot_bar(counts, labels, title, log_scale=False):
    unique_labels = np.unique(labels)
    countss = []
    for cypher in unique_labels:
        indices = labels == cypher
        cypher_data = counts[indices]
        x = np.mean(cypher_data)
        countss.append(x)
    y_pos = np.arange(len(unique_labels))
    if log_scale: countss = np.log(countss)
    plt.bar(y_pos, countss, align='center', alpha=0.5)
    plt.xticks(y_pos, unique_labels)
    if log_scale: plt.ylabel('log(count)')
    else: plt.ylabel('count')
    plt.title(title)
    plt.grid()
    plt.show()

if __name__ == '__main__':
    cyphertexts = np.load('./data/cyphertexts.npy')
    labels = np.load('./data/labels.npy')

    # letter_dist = count_letters(cyphertexts)
    # draw_distributions(letter_dist, labels)

    # adj_duplicates = count_adjacent_duplicates(cyphertexts)
    # plot_bar(adj_duplicates, labels, "Number of adjacent duplicates", log_scale=True)

    # bigrams = count_repeating_bigrams(cyphertexts)
    # no_rep_bigrams = bigrams[:,0]
    # most_freq_bigram_num = bigrams[:,1]
    # plot_bar(no_rep_bigrams, labels, "Number of repeating bigrams.")
    # plot_bar(most_freq_bigram_num, labels, "Frequency of the most frequent bigram.")

    # ios = index_of_coincidence(cyphertexts)
    # plot_bar(ios, labels, "Index of coincidence.")

    X = np.load('./data/X.npy')
    X = dim_reduction(X)
    plot_scatter_2d(X, labels)

