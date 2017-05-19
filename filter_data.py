import os
import numpy as np
import pickle
import argparse
import csv
import ujson

"""
Reads in clusters.csv files
Reads in original data file(s)
Searches each cluster of words & determines cluster word count for each sentences

RUN:

python filter_data.py "path/wordlists.list" "path/clusters.csv"

e.g. "../output-india-twitter/TC_saveFolder/wordlists.list" "../output-india-twitter/clusters.csv"
"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Determines relevancy of tweets & reddit comments input \n"
                                                 "Creates cluster association lists for each input")
    parser.add_argument('wordlistsDir', type=str, default='')
    parser.add_argument('clusterDir', type=str)
    args = parser.parse_args()

    return args


def load_cluster(clusterFile):
    clusters = []
    pathsList = []
    if os.path.isdir(clusterFile):
        for file in os.listdir(clusterFile):
            if os.path.splitext(file)[-1] == '.csv' or os.path.splitext(file)[-1] == '.tsv':
                pathsList.append(os.path.join(clusterFile, file))
    else:
        filePath = clusterFile.strip("[")
        filePath = filePath.strip("]")
        pathsList = (filePath).split(',')

    for path in pathsList:
        with open(path) as csvfile:
            clusterlist = csv.reader(csvfile)
            for row in clusterlist:
                clusters.append(row)
    return clusters


def main(dataFile, clusterFile):
    """
    :param dataFile: wordlists.list (output of text_clean.py - must have associated IDs)
    :param clusterFile: clusters.csv (output of text_clean.py)
    :return: clusters_freq_count.counts
    """
    print("loading files ...")
    clusters = load_cluster(clusterFile)
    wordlist = None
    if os.path.splitext(dataFile)[-1] == '.list':
        wordlist = pickle.load(open(dataFile, 'rb'))

    assert (isinstance(wordlist, dict))  # needs to be dict keyed with ids
    print("loaded files")
    num = len(wordlist)
    print("data has {} sentences".format(num))
    print("begin count")
    num_clusters = len(clusters)

    final_counts = {}
    progress = 0
    for post_id, sentences in wordlist.items():
        cluster_word_count = np.zeros((num_clusters,))
        for sentence in sentences:
            for word in sentence:
                for i in range(num_clusters):
                    if word in clusters[i]:
                        cluster_word_count[i] += 1

        sumcl = cluster_word_count.sum()
        progress += 1
        if sumcl != 0:
            cluster_word_count /= sumcl  # normalizes it
        if progress % 10000 == 0:
            print("completed counting: {:.2f}%".format((progress/num)*100))

        final_counts[post_id] = cluster_word_count

    print("100% complete - writing output file ... ")
    ujson.dump(final_counts, open(os.path.join(os.path.dirname(clusterFile), 'clusters_freq_count.counts'), 'w'))
    print("complete")


if __name__ == "__main__":
    args = parse_arguments()
    data = args.wordlistsDir
    cluster = args.clusterDir

    main(data, cluster)
