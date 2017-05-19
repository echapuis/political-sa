import os
import pandas as pd
import numpy as np
import argparse
import ujson
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

""" CALCULATES RELVANCY FOR EACH SUBMISSION & PLOTS RELEVANCY HISTOGRAM """ 

def parse_arguments():

    parser = argparse.ArgumentParser(description="reads in .counts files")
    parser.add_argument('cluster', type=str, default='')
    parser.add_argument('counts', type=str, default='')
    parser.add_argument('title', type=str, default='')
    parser.add_argument('data_file', type=str, default='')

    args = parser.parse_args()

    return args


def relevant_data(clusterFile, countFile, dataName):
    """
    :param clusterFile: cluster.csv
    :param countFile: clusters_freq_count.counts
    :return: list of relevant ids
    """

    count = ujson.loads(open(countFile).read())
    cluster = pd.read_csv(clusterFile, header=None, delimiter=",", usecols=[0])
    clusterlist = cluster[0].tolist()
    clusters_score = np.array(clusterlist)

    # dictionary
    cluster_relevance_all = {}
    list_hist = []
    for id, cluster_list in count.items():

        relevancy = np.array(cluster_list)
        count_relevancy = clusters_score*relevancy
        sum_rel = sum(count_relevancy)
        cluster_relevance_all[id] = sum_rel
        list_hist.append(sum_rel)

    ujson.dump(cluster_relevance_all, open(os.path.join(os.path.dirname(countFile) + '/' + dataName + '_rel.txt'), 'w'))
    return list_hist


def plot_histograms(data, title):

    val = np.array(data)
    plt.hist(val, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title(title)
    plt.xlabel('Percent Relevant')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":

    args = parse_arguments()
    cluster = args.cluster
    counts = args.counts
    title = args.title
    dataName = args.data_file

    list_hist = relevant_data(cluster, counts, dataName)
    plot_histograms(list_hist, title)
