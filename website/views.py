from flask import render_template
from flask import request
from website import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import random

import sys
sys.path.append('../')
import pun
from gensim.models.ldamodel import LdaModel as Lda
from gensim import corpora

wiki_topicmodel = Lda.load('models/180918_wikipedia_model.200.gensim.')

def convert_context_to_words(context, model=wiki_topicmodel):
    # Tokenize context
    context_tokens = pun.tokenize(context)
    # Convert to bow
    bag_of_words = wiki_topicmodel.id2word.doc2bow(context_tokens)

    topic_list = model.get_document_topics(bag_of_words)

    return get_words_from_top_topics(topic_list, wiki_topicmodel)



def get_words_from_top_topics(topic_list, model):

    list_of_words = []
    topic_list.sort(key=lambda tup: tup[1], reverse=True)

    for topic, topic_prob in topic_list:

        if topic_prob < 0.005:
            break

        for word_id, word_prob in model.get_topic_terms(topic, 500):
            if word_prob < 0.01:
                break
            list_of_words.append(model.id2word[word_id])

    return list_of_words


possible_pun_titles = ['Topical puns. On demand.',
                       'Your puns. Your pace.',
                       'Puns you can hardly understand.']

@app.route('/')

@app.route('/index', methods=['POST', 'GET'])
def index():
    error = None
    output_sentence = 'output here'
    input_sentence = None
    text_context = None
    if request.method == 'POST':
      # Do stuff with data that was posted
        input_sentence = request.form['input_sentence']
        text_context = request.form['context_text']

        words = convert_context_to_words(text_context)
        output_sentence = pun.insert_pun(input_sentence, words)



    return render_template("index.html",
                           title = random.choice(possible_pun_titles),
                           output_sentence = output_sentence,
                           input_sentence = input_sentence,
                           text_context = text_context)


