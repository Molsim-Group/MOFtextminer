import os

from pathlib import Path
from chemdataextractor.doc import Paragraph, Heading
from gensim.utils import SaveLoad, simple_preprocess
from gensim.corpora.dictionary import Dictionary
import numpy as np
import joblib
import pickle
import json

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from doc.utils import cleanup_text, split_text
from doc.storage import UnitStorage
from error import MerError

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Filter out INFO & WARNING messages


path_model_char = Path(__file__).parent / "libs/mer/keras/bilstmcrf_char"
rnn_model_with_char = load_model(str(path_model_char))

path_model_no_char = Path(__file__).parent / "libs/mer/keras/bilstmcrf"
rnn_model_without_char = load_model(str(path_model_no_char))

def import_bows(path):
    with open(path)) as f:
        data = json.load(f)
        bow_dictionary = Dictionary()
        bow_dictionary.merge_with(data)
    return bow_dictionary


def get_method_paragraphs(elements):
    abs_path = Path(__file__).parent
    bow_dictionary = import_bows(str(abs_path/"libs/paragraph_recognition/bow_dictionary.json"))
    log_reg = joblib.load(str(abs_path/"libs/paragraph_recognition/logreg_binary_method.sav"))
    
    method_paragraphs = []

    for i, element in enumerate(elements):
        if not isinstance(element, (Paragraph, Heading)):
            raise TypeError

        bows = bow_dictionary.doc2bow(simple_preprocess(element.text))
        
        matrix_ = np.zeros(shape=[1, 20038])
        for bow in bows:
            matrix_[0, bow[0]] = bow[1]

        if log_reg.predict(matrix_) == 1:
            before_element = elements[i-1]
            if isinstance(before_element, Heading):
                element_text = "<h> " + before_element.text + " </h> -end- " + element.text
            else:
                element_text = element.text
            method_paragraphs.append(element_text)

    if not method_paragraphs:
        raise MerError('There are no method paragraph')
    return method_paragraphs


def material_entity_recognition(method_paragraph, character_embedding=True, max_length=100):
    token_sents = _tokenize(method_paragraph)

    bio_tags = _get_bio_tags(character_embedding, *_get_pad_tokens(token_sents, character_embedding))
    
    return token_sents, bio_tags


def _tokenize(element):
    
    list_tokens = []

    text = cleanup_text(element)

    text = text.replace("dec -end-", 'decomposition')
    text = text.replace('decomp -end-', 'decomposition')
    text = text.replace('decomp. ) -end-', 'decomposition )')
    text = text.replace('dec. ) -end-', 'decomposition )')

    for sent in text.split("-end-"):
        if not sent:
            continue

        sent = UnitStorage.search_unit(sent)
        tokens = split_text(sent)

        if tokens:
            list_tokens.append(tokens)
    
    return list_tokens


def _get_pad_tokens(token_sents, character_embedding):
    # input : token_tokens -> 
    # return : list of padding map tokenized sentences, list of charactering padding map tokenized sentences
    # using pre-saved word2id, id2word, char2id

    word2id_path = Path(__file__).parent / "libs/mer/vocab/word2id"
    id2word_path = Path(__file__).parent / "libs/mer/vocab/id2word"
    
    with open(str(word2id_path), 'rb') as w2i:
        word2id = pickle.load(w2i)
    
    with open(str(id2word_path), 'rb') as i2w:
        id2word = pickle.load(i2w)
    
    # make pad_sequences with tokenized sentences
    map_token_sents = []
    for token_sent in token_sents:
        
        map_token_sent = list(map(lambda x: word2id[x] if x in word2id else 1, token_sent))
        map_token_sents.append(map_token_sent)
    
    pad_map_token_sents = pad_sequences(map_token_sents, padding="post", maxlen=100)  # [B, 100]

    if character_embedding: # make pad_sequences with tokenized words
        char2id_path = Path(__file__).parent / "libs/mer/vocab/char2id"
        with open(str(char2id_path), 'rb') as c2i:
            char2id = pickle.load(c2i)

        char_dim = pad_map_token_sents.shape
        pad_map_token_sents_char = np.empty([char_dim[0], char_dim[1], 30])  # [B, 100, 30]

        for i, sent in enumerate(pad_map_token_sents):
            map_sent = []
            for word in sent:
                map_word = list(map(lambda x: char2id[x] if x in char2id else 1, id2word[word]))
                if map_word == [100, 43, 28, 31, 101]:  # <PAD> -> 0
                    map_word = [0]
                map_sent.append(map_word)
            pad_map_token_sents_char[i] = pad_sequences(map_sent, padding="post", maxlen=30)
        return pad_map_token_sents, pad_map_token_sents_char

    else:
        return pad_map_token_sents


def _get_bio_tags(character_embedding, *args):
    """
    get bio tags from
    :param character_embedding:
    :param args: output of _get_bio_tags
    :return: (list) pad decoded tags
    """
    # model = rnn_model
    # x = pad_map_token_sents
    # x_char = pad_map_token_sents_char

    if character_embedding:
        model = rnn_model_with_char
        decode_tags, _, lens_text, _ = model(args)
    else:
        model = rnn_model_without_char
        decode_tags, _, lens_text, _ = model(args)
    
    # make zero in decode tags after lens_text
    pad_decode_tags = []
    for i, decode_tag in enumerate(decode_tags):
        mask = np.arange(100) < lens_text[i]
        decode_tag = np.where(mask, decode_tag, 0)
        pad_decode_tags.append(decode_tag)
    
    return pad_decode_tags
