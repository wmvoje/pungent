from flask import render_template
from flask import request
from website import app
from flask import redirect, url_for
# from sqlalchemy import create_engine
# from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import random
import csv


import sys
sys.path.append('../')
import pun
import web_format

from gensim import corpora

import os

if not os.path.exists('log.csv'):
    with open("log.csv", "w") as my_empty_csv:
        # now you have an empty file already
        pass  # or write something to it already

def append_row(csv_file_path, row):
    """This function was written to deal with having files left open"""
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)


# wiki_topicmodel = Lda.load('models/180918_wikipedia_model.200.gensim.')


log_file = 'log.csv'

###############################################################################
###############################################################################
########## MODEL FUNCTIONS



###############################################################################
###############################################################################

possible_pun_titles = ['Topical puns. On demand.',
                       'Your puns. Your pace.',
                       'Puns you can hardly understand.',
                       'Poorly generated puns for the people.']


@app.route('/')
@app.route('/index')

def index():
    ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    if request.args.get('title') is None:
        title = random.choice(possible_pun_titles)
    else:
        title = request.args.get('title')

    if request.args.get('context_text') is not None:
        context = request.args.get('context_text')
    else:
        context = None

    return render_template('index.html',
                            title = title,
                            text_context=context)


@app.route('/output', methods=['POST', 'GET'])
def output():
    # Get users IP address
    ip_address = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)

    if request.method == 'POST':
        user_selected = request.form['pun_selection']

        # Need to save this selection in a dataframe
        append_row(log_file, [ip_address, user_selected])

        # Add a new request which reloads a website
        # if request.form['pun_selection'] == 'None':
        return redirect(url_for('index', title="Let's try that PUNgain!",
                                 context_text=request.args.get('context_text')))


    context_text = request.args.get('context_text')
    input_sentence = request.args.get('input_sentence')
    ranked_substitutions, topic_words_considered = pun.generate_possible_pun_substitutions(context_text, input_sentence)

    # picking the top 5 puns
    log_output = [ip_address, context_text, input_sentence]

    list_of_puns = [web_format.html_bold_word(input_sentence, x) for x in
                    ranked_substitutions[:5]]
    # list_of_puns = [pun.substitute_pun(input_sentence, x[:3]) for x in
    #                 ranked_substitutions[:5]]

    output_sentence = list_of_puns
    log_output.extend(ranked_substitutions[:5])

    append_row(log_file, log_output)

    words_to_print = convert_topic_words_to_print(topic_words_considered)


    return render_template('output.html',
                           title = random.choice(possible_pun_titles),
                               output_sentence = list_of_puns,
                               input_sentence = input_sentence,
                               text_context = context_text,
                               considered_words = words_to_print)


def convert_topic_words_to_print(topic_words):
    output = [[]]
    topic_number = 1
    for topic in topic_words:
        if topic[1] == 'topicModel':
            return output
        if topic[2] != topic_number:
            topic_number = topic[2]
            output.append([topic[0]])
        output[topic_number-1].append(topic[0])

    return output
