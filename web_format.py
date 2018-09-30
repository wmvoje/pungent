#
from pun import split_text

def html_bold_word(sentence, word_substitions):

    pos_inds = []
    pos_alphas = []

    word_index = word_substitions[1]
    word = word_substitions[0]

    html_text = '<p>'
    for i,w in enumerate(split_text(sentence)):
        if i == word_index:
            w = word
        if i == 0:
            w = w.capitalize()
        else:
            if w in [',','!','?','.']:
                pass
            else:
                w = ' ' + w



        if i == word_index:
            html_text += "<span style='color: #000000; font-weight:bold;'>"+w+"</span>"
        else:
            html_text += w
    html_text += '</p>'
    return html_text
