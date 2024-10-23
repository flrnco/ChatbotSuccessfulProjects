# Libraries
import urllib.request
import collections
import os
import zipfile
import sys
import re
import pdb
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from faker import Faker
import random
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

TRAINING_MODE_g = "training"
TEST_MODE_g     = "test"

sys.stdout.reconfigure(encoding='utf-8')

# support functions
# Function to preprocess text and add spaces around punctuation
def preprocess_text(text):
    #print("1 - "+text)
    # Add spaces around punctuation (e.g., "word," becomes "word ,")
    text = re.sub(r'([!"#$%&()*+/:;<=>?@\[\\\]\'^_,`{|}~\t\n])', r' \1 ', text)
    #text = re.sub(r'([!,])', r' \1 ', text)
    #print("2 - "+text)
    # Remove any extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    #print("3 - "+text)
    return text
# Encode sentences thanks to a tokenizer
def encode(tokenizer_p, sentences_p):
    
    sequences = tokenizer_p.texts_to_sequences(sentences_p)
    return sequences
# Decode sentences thanks to a tokenizer
def decode(tokenizer_p,sequences_p):
    texts = tokenizer_p.sequences_to_texts(sequences_p)
    return texts
# Cut the sentences so that it has the appropriate length
def generate_subsequences(sentence_p, maxlen_p):
    subsequences = []
    if len(sentence_p) < maxlen_p:
        subsequences.append(sentence_p)
    else:
        for i in range(0, len(sentence_p)+int(0.4*maxlen_p)-int(maxlen_p), int(0.4*maxlen_p)):  # Sliding window
            subsequences.append(sentence_p[i:i + maxlen_p])
    return subsequences
####################################
### Positional encoding
####################################
def get_angles(pos, k, d):
    """
    Get the angles for the positional encoding
    
    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        k --   Row vector containing the dimension span [[0, 1, 2, ..., d-1]]
        d(integer) -- Encoding size
    
    Returns:
        angles -- (pos, d) numpy array 
    """
    
    # Get i from dimension span k
    i = k//2
    # Calculate the angles using pos, i and d
    angles = pos / 10000**(2*i/d)
    return angles
    
def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings 
    
    Arguments:
        positions (int) -- Maximum number of positions to be encoded ~ max size of the sentence captured
        d (int) -- Encoding size ~ dimension of the word embedding
    
    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    # START CODE HERE
    # initialize a matrix angle_rads of all the angles 
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d)[np.newaxis, :],
                            d)
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # END CODE HERE
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)

def map_indices_to_embeddings(tensor, reverse_words, word_to_vec_map):
    # Get the shape of the input tensor (batch_size, sequence_length)
    batch_size, sequence_length = tensor.shape

    # Create an empty array to hold the result (batch_size, sequence_length, embedding_dim)
    embeddings = np.zeros((batch_size, sequence_length, 50))

    # Iterate over each sentence and each word index
    for i in range(batch_size):
        for j in range(sequence_length):
            # Get the word index
            word_index = tensor[i, j].numpy()

            # Get the corresponding word using the reverse_words dictionary
            word = reverse_words.get(word_index, None)

            # If the word exists in the dictionary, fetch its embedding
            if word is not None:
                embedding = word_to_vec_map.get(word, np.zeros(50))  # Default to zeros if not found
            else:
                embedding = np.zeros(50)  # If the word isn't found, use a zero embedding

            # Assign the embedding to the output array
            embeddings[i, j] = embedding

    # Convert to TensorFlow tensor
    return tf.convert_to_tensor(embeddings, dtype=tf.float32)



##############################################################################################
#
# BusinessObjectsManager : store all the data structures that we will use during the run
#
#   Main attributes:
#      - problemName        : what we are solving
#      - ...
#
###############################################################################################
class BusinessObjectsManager(object):

    def __init__(self, problemName_p, nbWordsPerSentence_p = 25, oov_token_p = '<OOV>', filters_tokenizer_p = '#$%&/<=>@[\\]^_`{|}~\t\n'):
        """ Constructor """
        self._problemName                   = problemName_p
        self.nbWordsPerSentence             = nbWordsPerSentence_p
        
        # Reader Data structures
        ## Training set
        self.training_sentences             = []
        self.questions                      = []
        self.subjects                       = []
        self.action_verbs                   = []
        self.objectives                     = []
        self.indicesInTrainingSet           = []
        self.padded_sequences_training_set  = None      # will be the padded sequences of words translated into indices
        self.tensor_sequences_training_set  = None      # tensor built based on padded_sequences_training_set
        self.X_training                     = None      # tensor containing word embeddings representation of training set
        self.Y_training                     = None      # tensor containing all categorizations of the sentences from X_training
        
        ## Word embeddings
        self.words_in_vocabulary            = set()
        self.word_to_vec_map                = {}
        self.tokenizer                      = None      # will be a token later in the initialization
        self.oov_token                      = oov_token_p
        self.filters_tokenizer              = filters_tokenizer_p
        
        # Transformer model
        self.model                          = None      # will store the transformer tensor flow mode
        
    def __str__(self):
        ## Printor
        return 'myBOM'

    """ Class Functions """
    
        
    
    """ ========= GETTERS and SETTERS =========== """
    """ Problem Name """
    def _get_problemName(self):
        return self._list_Days_Study
    def _set_problemName(self, problemName_l):
        self._problemName = problemName_l
    problemName = property(_get_problemName, _set_problemName)