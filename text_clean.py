import os
from os.path import join
import time
import pandas as pd
import numpy as np
import pickle
import warnings
import argparse
import logging
import csv
import ujson
import sys

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk.data

from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import KMeans
import scipy.spatial.distance as dist

warnings.simplefilter("ignore")
output_dir = 'output'
save_dir = 'TC_saveFolder'
input_file = ''
load = True
save = True
verbose = False
np.random.seed(0)

""" RUNS WORD2VEC & CLUSTERING - edit parameters to train Word2Vec in train_Word2Vec """

# Helper Functions
def parse_arguments():
    parser = argparse.ArgumentParser(description="Parameters specify where/how files are saved. \n"
                                                 "This program takes as input a csv,tsv, or json file and outputs stuff.")
    parser.add_argument('input_file', type=str, default=input_file)
    parser.add_argument('output_dir', type=str, default=output_dir)
    parser.add_argument('--save', '-s', action='store_true', default=True,
                        help='saves intermediary files to save_dir')
    parser.add_argument('--load', '-l', action='store_true', default=True,
                        help='loads intermediary files automatically if available')
    parser.add_argument('--fresh', '-f', action='store_true', default=False,
                        help='sets loading and saving to false, overwriting existing files')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='prints program progress')
    parser.add_argument('--save_dir', type=str, default=join(output_dir, save_dir))
    args = parser.parse_args()
    if args.fresh:
        args.save = False
        args.load = False
    return args


def mkdir(dirPath):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    else:
        print("WARNING: Directory {} already exists. Data may be overwritten if 'load' option is disabled.".format(
            dirPath), flush=True)
        if not load:
            print("You have 3 seconds to terminate program...", flush=True)
            time.sleep(3)


def text_to_wordlist(text, remove_stopwords=False, stemming=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.

    # 1. Remove HTML
    review_text = BeautifulSoup(text).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("'", "", review_text)
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Optionally stem topically similar words
    if stemming:
        p_stemmer = PorterStemmer()
        words = [p_stemmer.stem(w) for w in words]

    return [words]


def text_to_sentences(text, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    # raw_sentences = tokenizer.tokenize(review.decode('utf-8').strip())
    raw_sentences = tokenizer.tokenize(str(text).strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences += [text_to_wordlist(raw_sentence, remove_stopwords=remove_stopwords)]
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def newprint(string):
    print(string, flush=True)
    sys.stdout.flush()


def train_Word2Vec(sentences, saveAs=''):
    num_features = 300  # Word vector dimensionality
    min_word_count = max(20.0, len(sentences) * 0.00005)  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-4  # Downsample setting for frequent words

    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1, )

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()

    if save:
        model.save(join(output_dir, save_dir, 'w2v_Model.w2v'))

    dictionary = KeyedVector_to_Dict(model.wv, saveAs=saveAs)

    return dictionary


def KeyedVector_to_Dict(kv, saveAs=''):
    words = set(kv.index2word)
    dict = {}
    for word in words:
        dict[word] = kv[word]

    if saveAs != '' and save:
        pickle.dump(dict, open(saveAs, 'wb'))

    return dict


def load_GoogleWords(words, saveAs=''):
    firstStart = time.time()
    if verbose: newprint("Loading Google Word2Vec...")
    start = time.time()
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    if verbose: newprint("Finished loading in {} seconds. Generating Dictionary...".format(time.time() - start))
    start = time.time()
    dictionary = {}
    missing_words = 0
    for word in words:
        try:
            dictionary[word] = model[word]
        except KeyError:
            missing_words += 1
            # print("Word {} not in Google's Word2Vec dictionary.".format(word))

    if verbose: newprint(
        "Finished generating dict in {} seconds.\n {} words were not found in Word2Vec dictionary. This usually includes people names and typos.".format(
            time.time() - start, missing_words))
    if saveAs != '' and save:
        if verbose: newprint("Saving Model...")
        pickle.dump(dictionary, open(saveAs, 'wb'))

    if verbose: newprint("Finished loading words from Google Word2Vec in {} seconds.".format(time.time() - firstStart))
    return dictionary


def Dict_to_Matrix(dict, saveAs=''):
    items = sorted(dict.items())
    N = len(items)
    M = len(items[0][1])
    matrix = np.zeros((N, M))

    for i in range(N):
        matrix[i, :] = items[i][1]

    if saveAs != '':
        pickle.dump(matrix, open(saveAs, 'wb'))

    return matrix


def sort_clusters(clusters, wv, centroids):
    sorted_clusters = []
    for i in range(len(clusters)):
        distances = []
        for word in clusters[i]:
            distances += [dist.euclidean(wv[word], centroids[i])]
        sorted_clusters.append([words for (dists, words) in sorted(zip(distances, clusters[i]))])
    return sorted_clusters


def save_Cluster(cluster_list, saveAs):
    with open(saveAs, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(cluster_list)


def get_bagOfCentroids(reviews, word_centroid_map):
    num_reviews = len(reviews)
    num_centroids = max(word_centroid_map.values()) + 1

    bag_matrix = np.zeros((num_reviews, num_centroids), dtype='float32')

    for i in range(num_reviews):
        for word in reviews[i]:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_matrix[i, index] += 1

    return bag_matrix


def load_file(filePath):
    # file = pd.read_csv()
    fname, ext = os.path.splitext(filePath)

    dictionary = {}
    if ext == '.json':
        data = ujson.loads(open(filePath).read())

        for d1 in data:
            sid = d1.get('SubmissionID')
            dictionary[sid] = d1.get('SubmissionTitle')
            com = d1.get("Comments")
            for d2 in com:
                cid = d2.get("CommentID")
                dictionary[cid] = d2.get('CommentText')

    elif ext == '.csv' or ext == '.tsv':
        data = pd.read_csv(filePath, header=0, index_col=[], delimiter=",", quoting=1, encoding='latin1')
        for row in data.itertuples():
            if (not (pd.isnull(row.id) or pd.isnull(row.text))):
                dictionary[row.id] = row.text

    else:
        pathsList = []
        if os.path.isdir(filePath):
            for file in os.listdir(filePath):
                if os.path.splitext(file)[-1] == '.csv' or os.path.splitext(file)[-1] == '.tsv' or os.path.splitext(file)[-1] == '.txt':
                    pathsList.append(os.path.join(filePath, file))
        else:
            filePath = filePath.strip("[")
            filePath = filePath.strip("]")
            pathsList = (filePath).split(',')

        for path in pathsList:
            data = pd.read_table(path, header=0, delimiter="\t", encoding='latin1')
            for row in data.itertuples():
                if (not (pd.isnull(row[1]) or pd.isnull(row[2]))):
                    dictionary[row[1]] = row[2]

    return dictionary

def get_CleanText(data, type, saveAs='', remove_stopwords=False):
    if load:
        if type == 's':
            if os.path.exists(saveAs):
                if verbose: newprint("Loaded sentences.")
                return pickle.load(open(saveAs, 'rb'))
        if type == 'w':
            if os.path.exists(saveAs):
                if verbose: newprint("Loaded wordlists.")
                return pickle.load(open(saveAs, 'rb'))

    if verbose and type == 'w': newprint("Converting to wordlists...")
    if verbose and type == 's': newprint("Converting to sentences...")

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    num_reviews = len(data)
    start = time.time()
    i = 0
    z = 0
    ten = num_reviews / 10
    if isinstance(data, dict):
        clean_data = {}  # Initialize an empty dict of sentences which is keyed with the id
        for id, review in data.items():
            if type == 's': clean_data[id] = text_to_sentences(review, tokenizer, remove_stopwords=remove_stopwords)
            if type == 'w': clean_data[id] = text_to_wordlist(review, remove_stopwords=remove_stopwords)
            i += 1
            if i % ten == 0:
                z += 1
                if verbose: newprint("\r{}% of text reformatted after {} seconds.".format(z * 10, time.time() - start))

    else:
        clean_data = []  # Initialize an empty list of sentences
        for review in data:
            if type == 's': clean_data += text_to_sentences(review, tokenizer, remove_stopwords=remove_stopwords)
            if type == 'w': clean_data += text_to_wordlist(review, remove_stopwords=remove_stopwords)
            i += 1
            if i % ten == 0:
                z += 1
                if verbose: newprint("\r{}% of text reformatted after {} seconds.".format(z * 10, time.time() - start))

    if saveAs != '' and save:
        pickle.dump(clean_data, open(saveAs, 'wb'))

    return clean_data


def get_WordVecs(sentences, saveAs='', pretrained=False):
    if load:
        if os.path.exists(saveAs):
            if verbose: newprint("Loaded word vectors.")
            return pickle.load(open(saveAs, 'rb'))

    if verbose: newprint("Calculating word vectors...")

    if isinstance(sentences, dict):
        sentence_list = []
        for sentence in sentences.values():
            for word in sentence:
                sentence_list += word
        sentences = sentence_list

    if pretrained:
        words = []
        for sentence in sentences:
            for word in sentence:
                words += word
        words = set(words)
        dictionary = load_GoogleWords(words, saveAs=saveAs)
    else:
        dictionary = train_Word2Vec(sentences, saveAs=saveAs)

    return dictionary


def get_Clusters(word_vectors, saveAs='', n_clusters=100):
    skip_Kmeans = False
    skip_wcm = False
    if load:
        if os.path.exists(saveAs):
            if verbose: newprint("Loaded cluster map.")
            word_centroid_map = pickle.load(open(saveAs, 'rb'))
            if os.path.exists(join(output_dir, 'clusters.csv')):
                return word_centroid_map
            skip_wcm = True

        path = join(output_dir, save_dir, 'KMeans_Model.skl')
        if os.path.exists(path):
            kmeans = pickle.load(open(path, 'rb'))
            skip_Kmeans = True
            if verbose: newprint("Loaded kMeans model.")

    if verbose: newprint("Calculating word clusters...")

    keys = sorted(word_vectors.keys())
    matrix = Dict_to_Matrix(word_vectors)

    if not skip_Kmeans:
        k = n_clusters
        # Initalize a k-means object and use it to extract centroids
        if verbose: newprint("Running K means. This may take a few minutes.")
        start = time.time()
        kmeans = KMeans(n_clusters=k, verbose=0)
        kmeans.fit(matrix)
        if verbose: newprint("Finished K means in {:.0f}:{:.0f} minutes.".format((time.time() - start) // 60,
                                                                                 (time.time() - start) % 60))
        if save:
            pickle.dump(kmeans, open(join(output_dir, save_dir, 'Kmeans_Model.skl'), 'wb'))

    if not skip_wcm:
        idx = kmeans.predict(matrix)
        word_centroid_map = dict(zip(keys, idx))
        if saveAs != '' and save:
            pickle.dump(word_centroid_map, open(saveAs, 'wb'))

    # Print the first ten clusters
    all_words = []
    for cluster in range(n_clusters):
        # Find all of the words for that cluster number, and print them out
        words = []
        items = list(word_centroid_map.items())
        for i in range(len(items)):
            if (items[i][1] == cluster):
                words.append(items[i][0])

        all_words.append(words)

    if verbose: newprint("Sorting the clusters...")
    clusters = sort_clusters(all_words, word_vectors, kmeans.cluster_centers_)

    save_Cluster(clusters, saveAs=join(output_dir, 'clusters.csv'))

    return word_centroid_map


def get_BOC(ids, wordlists, word_centroid_map, saveAs=''):
    if load:
        if os.path.exists(saveAs):
            if verbose: newprint("Loaded bag of centroids.")
            return pickle.load(open(saveAs, 'rb'))

    word_lists = []
    ids_ = []
    if isinstance(wordlists, dict):
        for id, wordlist in wordlists.items():
            ids_.append(id)
            word_lists += wordlist
    wordlists = word_lists
    ids = ids_

    if verbose: newprint("Calculating bag of centroids...")
    num_wordlist = len(wordlists)
    num_centroids = max(word_centroid_map.values()) + 1

    bag_matrix = np.zeros((num_wordlist, num_centroids)).astype(np.float32)
    for i in range(num_wordlist):
        sum = 0
        for word in wordlists[i]:
            if word in word_centroid_map:
                index = word_centroid_map[word]
                bag_matrix[i, index] += 1
                sum += 1
        bag_matrix[i, :] = bag_matrix[i, :] / sum

    dictionary = {}
    for i in range(bag_matrix.shape[0]):
        dictionary[ids[i]] = bag_matrix[i, :]

    if saveAs != '':
        pickle.dump(dictionary, open(saveAs, 'wb'))
    print("Bag of centroids computed")
    return dictionary


if __name__ == "__main__":
    args = parse_arguments()
    output_dir = args.output_dir
    save = args.save
    load = args.load
    verbose = args.verbose
    input_file = args.input_file

    mkdir(output_dir)
    if save:
        mkdir(join(output_dir, save_dir))

    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Loads the text
    # Output: dict {id:text}
    if verbose:
        newprint("Loading file...")
    filtered_text = load_file(input_file)
    if verbose:
        newprint("Number of text entries read: {}".format(len(filtered_text)))

    # Filters out all non-English text
    # Output: dict {id:text}, saveFile: '<output_dir>/<save_dir>/filtered_text.dict'
    # filtered_text = language_filter(data)
    ids = list(filtered_text.keys())
    text = list(filtered_text.values())

    # Converts text to wordlists for BOC
    # Output: list[wordlist], saveFile: '<output_dir>/<save_dir>/wordlists.list'
    # wordlists = get_CleanText(text, type='w', saveAs=join(output_dir, save_dir, 'wordlists.list'))
    # Output: dict{id:wordlist}, saveFile: '<output_dir>/<save_dir>/wordlists.list'
    wordlists = get_CleanText(filtered_text, type='w', saveAs=join(output_dir, save_dir, 'wordlists.list'))

    if verbose:
        newprint("Length of wordlist: {}".format(len(wordlists)))

    # Converts text to sentences to get Word Vectors
    # Output: list[sentences], saveFile: '<output_dir>/<save_dir>/sentences.list'
    # sentences = get_CleanText(text, type='s', saveAs=join(output_dir, save_dir, 'sentences.list'))
    # Output: dict{id:wordlist}, saveFile: '<output_dir>/<save_dir>/wordlists.list'
    sentences = get_CleanText(filtered_text, type='s', saveAs=join(output_dir, save_dir, 'sentences.list'))
    if verbose:
        newprint("Length of sentences: {}".format(len(sentences)))

    # Calculates word_vectors using word2vec
    # Output: dict {word:vector}, saveFile: '<output_dir>/<save_dir>/word_vectors.dict'
    # Note: if save parameter is on, trained word2vec models will be stored in the save directory
    word_vectors = get_WordVecs(sentences, pretrained=False, saveAs=join(output_dir, save_dir, 'word_vectors.dict'))
    if verbose:
        newprint("Length of word_vectors: {}".format(len(word_vectors)))

    # Calculates word clusters using KMeans
    # Output: dict {word:cluster}, saveFile: '<output_dir>/<save_dir>/cluster_map.dict'
    # Note: clusters are saved by default in '<output_dir>/clusters.csv'
    cluster_map = get_Clusters(word_vectors, saveAs=join(output_dir, save_dir, 'cluster_map.dict'))
    if verbose:
        newprint("Length of cluster_map: {}".format(len(cluster_map)))

    # Converts wordlists to bag of centroids
    # Output: dict {id:BOC}
    # Note: bag of centroids are saved by default in '<output_dir>/bag_of_centroids.dict'
    bag_of_centroids = get_BOC(ids, wordlists, cluster_map, saveAs=join(output_dir, 'bag_of_centroids.dict'))
    if verbose:
        newprint("Shape of bag of centroids: {}:{}".format(len(bag_of_centroids), len(bag_of_centroids[ids[0]])))
