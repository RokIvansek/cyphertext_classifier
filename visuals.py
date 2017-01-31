import sys
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def draw_distributions(dist_of_letters, labels):
    for cypher in np.unique(labels):
        indices = labels == cypher
        cypher_data = dist_of_letters[indices, :]
        x = np.mean(cypher_data, axis=0).astype(int)
        objects = (list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, x, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Count')
        plt.title(cypher)
        plt.show()

def plot_no_adj_duplicates(counts, labels):
    unique_labels = np.unique(labels)
    no_adj_dups = []
    for cypher in unique_labels:
        indices = labels == cypher
        cypher_data = counts[indices]
        x = np.mean(cypher_data)
        no_adj_dups.append(x)
    y_pos = np.arange(len(unique_labels))
    plt.bar(y_pos, np.log(no_adj_dups), align='center', alpha=0.5)
    plt.xticks(y_pos, unique_labels)
    plt.ylabel('log(count)')
    plt.title('Number of adjacent duplicates')
    plt.show()

if __name__ == '__main__':
    X = np.load(sys.argv[1])
    labels = np.load(sys.argv[2])
    # letter_dist = X[:, :-1]
    # draw_distributions(letter_dist, labels)
    counts = X[:,-1]
    plot_no_adj_duplicates(counts, labels)