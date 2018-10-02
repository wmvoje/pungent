""" Collection of functions to make a pun
"""
# Imports

import nltk
from random import shuffle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk import pos_tag
import enchant
import pickle
from gensim.models import doc2vec
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel as Lda
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import re



spelling_dict = enchant.Dict("en_US")
stemmer = PorterStemmer()

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

phone_dict = nltk.corpus.cmudict.dict()

# load in models and features

wiki_topicmodel = Lda.load('models/180925_wikipedia_model.individually_binned.200.gensim.')

# loading the stemmed_dict
with open('180922_stemmed_dict.p', 'rb') as tounpick:
    stemmed_dict = pickle.load(tounpick)

# Loading the doc2vec
wiki_doc2vec = doc2vec.Doc2Vec.load('models/simple_wiki_chunked_doc2vec_300_vector_10_min_word')

# loading the doc2vec corpus
with open('models/simple_wiki_chunked_corpus_10_count_cutoff.p', 'rb') as tounpcik:
    wiki_doc2vec_corpus = pickle.load(tounpcik)

# loading the TF-IDF information
tf_idf_dict = Dictionary.load('models/TF-IDF_dict.D')

# loading the TF-IDF model
tf_idf_model = TfidfModel.load('models/TF-IDF_model')

# loading the corpus
with open('models/TF-IDF_corpus.p', 'rb') as tounpick:
    tf_idf_corpus = pickle.load(tounpick)

### Functions for generating puns


def sentence_to_doc2vec(text, model):
    """
    Iterator which spits out words that are found in
    the most 'topical' textual elements
    """
    # parse the sentence
    text = simple_preprocess(text)
    # Find the respective doc2vec vector
    text_vector = model.infer_vector(text)
    # find the most similar text pieces
    most_similar_documents_with_score = model.docvecs.most_similar([text_vector])

    for document_id, cosine_sim_score in most_similar_documents_with_score:

        ## Calculate the TF-IDF for the document

        words_with_tf_score = tf_idf_of_document(document_id, tf_idf_model,
                                                 tf_idf_dict, tf_idf_corpus,
                                                 tf_idf_cutoff=0.08)



        yield (words_with_tf_score, cosine_sim_score)


def tf_idf_of_document(document_id, model, dictionary, corpus, tf_idf_cutoff=0.07):

    # get the tf idf from the document id
    tf_idf = model[corpus[document_id]]
#     tf_idf.sort(key=lambda x: x[1])

    output = []
    tf_idf_value = None
    for index, tf_idf_value in tf_idf:
        if tf_idf_value > tf_idf_cutoff:
            output.append((dictionary.id2token[index], tf_idf_value))

    return output



def split_text(string):
    return re.findall(r"[\w']+|[.,!?;]", string)



def generate_possible_pun_substitutions(context, input_sentence, w2v_number=10):
    """
    Takes context and input sentence

    returns list of possible substitutions with scores and the topic
    words consdiered

    """

    # First process context
    doc2vec_word_generator = sentence_to_doc2vec(context, wiki_doc2vec)
    topic_words, topic_score = sentence_to_topicmodel_words(context, wiki_topicmodel)

    # Then try to generate sentences using these metrics
    output = []
    topic_words_considered = []

    # consider word2vec words
    list_of_words = []
    for i in range(w2v_number):
        words, w2v_score = next(doc2vec_word_generator)

        # incoperate TF-IDF information


        topic_words_considered.extend([[word, 'doc2vec', i+1, tf_score*w2v_score, w2v_score, tf_score]
                                        for word, tf_score in words])


    #### REMOVE DUPLICATE WORDS
    list_of_words = []
    topic_words_filtered = []
    for out in topic_words_considered:
        if out[0] in list_of_words:
            continue
        else:
            list_of_words.append(out[0])
            topic_words_filtered.append(out)


    # generate all of the substituions
    sub_tuples = enumerate_PD_pun_subs_w2v(input_sentence, topic_words_filtered)
    # word, sub_index, phonetic_distance

    for word, sub_index, phon_dist, score, group in sub_tuples:
        output.append([word, sub_index, phon_dist, 'doc2vec',
                       group, score, score/phon_dist])



    # Now do topic words
    # sub_tuples = enumerate_PD_pun_subs(input_sentence, topic_words)
    # for word, sub_index, phon_dist in sub_tuples:

    #     topic_words_considered.append([word, 'topicModel', 1, topic_score])

    #     output.append([word, sub_index, phon_dist, 'topicModel', 1, topic_score, phon_dist/topic_score])

    output.sort(key=lambda x: x[6])

    output.reverse()

    return output, topic_words_filtered


def enumerate_PD_pun_subs_w2v(sentence, possible_words_with_score,
                              max_distance=4, max_return=10):
    """
    Takes a sentence and possible words and creates returns an array of possible
    pun substituions based on phonetic distance
    """
    output = []
    sentence_words = list(split_text(sentence))

    for word_index, word in enumerate(sentence_words):
        for word_vec in possible_words_with_score:
            pos_word = word_vec[0]
            group = word_vec[2]
            score = word_vec[3]

            if pos_word in word:
                # This substituion would be meaningless
                continue

            dist = phonetic_distance(word, pos_word)
            if dist <= max_distance:
                # Decrease the distance
                output.append((pos_word, word_index, dist, score, group))
    # output.sort(key=lambda tup: tup[2])
    return output


def get_words_from_top_topic(topic_list, model, min_word_prob=0.05):
    """
    First finding all of the words
    """

    list_of_words = []
    topic_list.sort(key=lambda tup: tup[1], reverse=True)

    topic, topic_prob = topic_list[0]

    for word_id, word_prob in model.get_topic_terms(topic, 100):
        if word_prob < min_word_prob:
            break
        if model.id2word[word_id] in stemmed_dict:
            for word in stemmed_dict[model.id2word[word_id]]:
                list_of_words.append(word)
        else:
            list_of_words.append(model.id2word[word_id])


    return list_of_words, topic_prob


def sentence_to_topicmodel_words(sentence, model):
    """
    Returns a list of words and the rank of the topic model
    """
    context_tokens = tokenize(sentence)
    bag_of_words = model.id2word.doc2bow(context_tokens)
    document_topics = model.get_document_topics(bag_of_words)
    return get_words_from_top_topic(document_topics, model)




def enumerate_PD_pun_subs(sentence, possible_words, max_distance=4, max_return=10):
    """
    Takes a sentence and possible words and creates returns an array of possible
    pun substituions based on phonetic distance
    """
    output = []
    sentence_words = list(split_text(sentence))
    for word_index, word in enumerate(sentence_words):
        for pos_word in possible_words:
            if pos_word in word:
                # This substituion would be meaningless
                continue

            dist = phonetic_distance(word, pos_word)
            if dist <= max_distance:
                # Decrease the distance
                output.append((pos_word, word_index, dist))
    output.sort(key=lambda tup: tup[2])
    return output


def substitute_pun(sentence, sub_tuple):
    """Takes a sentence
    and a touple of (word, index, and score)
    and makes a sentence
    """
    sentence_words = list(split_text(sentence))
    sentence_words[sub_tuple[1]] = sub_tuple[0]
    return ' '.join(word for word in sentence_words)



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
            word in split_text(sentence.lower())]

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
                         'TO', 'RP', 'RBS', 'RB', 'RBR', 'PRP$', 'PRP',
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


### Depreciated probably #####

def insert_pun(sentence, possible_words, max_distance = 2):
    best_distance = max_distance
    best_index = None
    best_word = None
    sentence_words = list(split_text(sentence))
    for word_index, word in enumerate(sentence_words):
        # shuffle(possible_words)
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
