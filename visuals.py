import sys
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from get_features import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
import matplotlib.patches as mpatches

def subsample(X, labels, sample_size=1):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    _, X_sub, _, labels_sub = train_test_split(X, labels, test_size=sample_size)
    return X_sub, le.inverse_transform(labels_sub)

def dim_reduction(X):
    ss = StandardScaler()
    pca = PCA(n_components=2)
    X = pca.fit_transform(ss.fit_transform(X))
    return X

def mds(X):
    ss = StandardScaler()
    mds = MDS(n_jobs=-1)
    X = mds.fit_transform(ss.fit_transform(X))
    return X

def plot_scatter_2d(X, labels):
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    unique_num_labels = np.unique(labels)
    unique_labels = le.inverse_transform(unique_num_labels)
    print(unique_num_labels)
    print(unique_labels)
    colors=['b','g','r','c','m','y']
    plt.scatter(X[:,0], X[:,1], s=75, c=[colors[i] for i in labels], alpha=0.5)
    # patches = mpatches.Patch(color=all_lables, label=le.inverse_transform(all_lables))
    plt.legend(handles=[mpatches.Patch(color=colors[i], label=unique_labels[i]) for i in unique_num_labels])
    plt.legend()
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

def plot_accuracy_score():
    labels = [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000]
    accuracy = [0.829378317614, 0.862716103369, 0.876875448497, 0.888757423716, 0.893335566778,
                0.896674552443, 0.893135477295, 0.89605345238, 0.893343807012]
    y_pos = np.arange(len(labels))
    plt.xticks(y_pos, labels)
    plt.plot(accuracy)
    plt.grid()
    plt.ylabel('Percantage of samples correctly classified')
    plt.xlabel('Number of descision trees')
    plt.title('Accuracy score')
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

    # X = np.load('./data/X.npy')
    # X, labels = subsample(X, labels, sample_size=0.1)
    #
    # X = dim_reduction(X)
    # X = mds(X)
    # plot_scatter_2d(X, labels)

    plot_accuracy_score()

