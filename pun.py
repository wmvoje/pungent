""" Collection of functions to make a pun
"""
# Imports

import cmudict
import nltk
from random import shuffle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag
import enchant
spelling_dict = enchant.Dict("en_US")
stemmer = PorterStemmer()

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

phone_dict = nltk.corpus.cmudict.dict()


def insert_pun(sentence, possible_words, max_distance = 2):
    best_distance = max_distance
    best_index = None
    best_word = None
    sentence_words = list(sentence.split())
    for word_index, word in enumerate(sentence_words):
        shuffle(possible_words)
        for pos_word in possible_words:
            if pos_word in word:
                # This substituion would be meaningless
                continue
            dist = phonetic_distance(word, pos_word)
#
            if dist <= best_distance:
                # Decrease the distance
                best_distance += -1
                best_index = word_index
                best_word = pos_word

    if best_word is None:
        return 'Sorry, no pun found!'

    sentence_words[best_index] = best_word

    return ' '.join(word for word in sentence_words)

# extracting phones from words and sentences

# consider using metaphonedoble instead of this library

def word_to_phoneme(word):
    """Converts a word to a list of phones"""
    try:
        return phone_dict[word][0]
    except:
        return None

def sentence_to_word_of_phoneme(sentence):
    """takes string sentence and returns
    list of lists of composing phones"""
    return [word_to_phoneme(word) for
            word in sentence.lower().split()]

def subfinder_bool(mylist, pattern):
    """if a subpattern is in a list return
    a bool"""
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append(pattern)
            return True
    return False

# phone comparisions

def edit_distance(w1, w2):
    """Code taken from
    https://github.com/maxwell-schwartz/PUNchlineGenerator

    Levenshtein distance modified such that deletions and addition are cost 2
    and deletions cost 1
    """

    cost = []

    if (w1 is None) or (w2 is None):
        # Return a number that' huge
        return 100

    for i in range(len(w1)+1):
        x = []
        for j in range(len(w2)+1):
            x.append(0)
        cost.append(x)

    for i in range(len(w1)+1):
        cost[i][0] = i
    for j in range(len(w2)+1):
        cost[0][j] = j

    # baseline costs
    del_cost = 2
    add_cost = 2
    sub_cost = 1

    for i in range(1, len(w1)+1):
        for j in range(1, len(w2)+1):
            if w1[i-1] == w2[j-1]:
                sub_cost = 0
            else:
                sub_cost = 2
            # get the totals
            del_total = cost[i-1][j] + del_cost
            add_total = cost[i][j-1] + add_cost
            sub_total = cost[i-1][j-1] + sub_cost
            # choose the lowest cost from the options
            options = [del_total, add_total, sub_total]
            options.sort()
            cost[i][j] = options[0]

    return cost[-1][-1]

def phonetic_distance(word1, word2):
    """compares two words and returns phonetic
    distance"""
    phoneme1 = word_to_phoneme(word1.lower())
    phoneme2 = word_to_phoneme(word2.lower())

    return edit_distance(phoneme1, phoneme2)


import os
import random
import codecs
from collections import defaultdict

from gensim.models.ldamodel import LdaModel as Lda
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag

import enchant
spelling_dict = enchant.Dict("en_US")

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
stemmer = PorterStemmer()

stemmed_dict = defaultdict(set)

def stem_and_update_stem_dict(tokens):
    output_list = []
    for token in tokens:
        stemmed = stemmer.stem(token)
        if stemmed != token:
            stemmed_dict[stemmed].add(token)
        output_list.append(stemmed)
    return output_list

list_of_POS_to_ignore = ['WRB', 'WP$', 'WP',  'WDT', 'UH',
                         'TO', 'RP', 'RBS', 'RBR', 'PRP$', 'PRP',
                        'MD', 'JJS', 'JJR', 'JJ', 'IN', 'FW', 'EX',
                         'DT', 'CD']

# Function to remove stop words from sentences & lemmatize verbs.
def tokenize(doc, stem=True, initial_word_split=True):
    if initial_word_split:
        tokens = word_tokenize(doc)
    else:
        tokens = doc
    #removing stop words
    tokens = [i for i in tokens if i not in stop]
    # removing pos data
    tokens = [word for word, pos in pos_tag(tokens) if pos not in list_of_POS_to_ignore]
    # Removing improperly spelled words (pronouns must be capitalized to be spelled right)
    tokens = [word for word in tokens if spelling_dict.check(word)]
    # lowercase
    tokens = [word.lower() for word in tokens]
    # lemmatized
    tokens = [lemma.lemmatize(word, 'v') for word in tokens]
    # removing short words
    tokens = [s for s in tokens if len(s) > 2]
    # stemmed
    if stem:
        tokens = [stemmer.stem(s) for s in tokens]

    return tokens



list_of_POS_to_ignore = ['WRB', 'WP$', 'WP',  'WDT', 'UH',
                         'TO', 'RP', 'RBS', 'RBR', 'PRP$', 'PRP',
                        'MD', 'JJS', 'JJR', 'JJ', 'IN', 'FW', 'EX',
                         'DT', 'CD']

