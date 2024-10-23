from AI_chat_engine import BOM
#import UI
from datetime import datetime, timedelta
import operator
import time
import sys
import math
from operator import attrgetter
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Activation, Lambda, Embedding
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import numpy as np
import spacy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # remove warnings due to compatibility v1 / v2

nbCallEncoderSingleLayer_g = 0
ID_EncoderSingleLayer_g = 0
ID_EncoderLayer_g = 0

# ## Masking
# 
# There are two types of masks that are useful when building your Transformer network: the *padding mask* and the *look-ahead mask*. Both help the softmax computation give the appropriate weights to the words in your input sentence. 
# 
# ### 2.1 - Padding Mask
# 
# Oftentimes your input sequence will exceed the maximum length of a sequence your network can process. Let's say the maximum length of your model is five, it is fed the following sequences:
# 
#     [["Do", "you", "know", "when", "Jane", "is", "going", "to", "visit", "Africa"], 
#      ["Jane", "visits", "Africa", "in", "September" ],
#      ["Exciting", "!"]
#     ]
# 
# which might get vectorized as:
# 
#     [[ 71, 121, 4, 56, 99, 2344, 345, 1284, 15],
#      [ 56, 1285, 15, 181, 545],
#      [ 87, 600]
#     ]
#     
# When passing sequences into a transformer model, it is important that they are of uniform length. You can achieve this by padding the sequence with zeros, and truncating sentences that exceed the maximum length of your model:
# 
#     [[ 71, 121, 4, 56, 99],
#      [ 2344, 345, 1284, 15, 0],
#      [ 56, 1285, 15, 181, 545],
#      [ 87, 600, 0, 0, 0],
#     ]
#     
# Sequences longer than the maximum length of five will be truncated, and zeros will be added to the truncated sequence to achieve uniform length. Similarly, for sequences shorter than the maximum length, zeros will also be added for padding. However, these zeros will affect the softmax calculation - this is when a padding mask comes in handy! You will need to define a boolean mask that specifies to which elements you must attend(1) and which elements you must ignore(0). Later you will use that mask to set all the zeros in the sequence to a value close to negative infinity (-1e9). 
# 
# After masking, your input should go from `[87, 600, 0, 0, 0]` to `[87, 600, -1e9, -1e9, -1e9]`, so that when you take the softmax, the zeros don't affect the score.
# 
# The [MultiheadAttention](https://keras.io/api/layers/attention_layers/multi_head_attention/) layer implemented in Keras, uses this masking logic.
# 
# **Note:** The below function only creates the mask of an _already padded sequence_. Later in this week, you’ll go through some Labs on Transformer applications, where you’ll be introduced to [TensorFlow Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) and [Hugging Face Tokenizer](https://huggingface.co/docs/tokenizers/api/tokenizer), which internally handle padding (and truncating) the input sequence.

def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids -- (n, m) matrix
    
    Returns:
        mask -- (n, 1, m) binary tensor
    """    
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
  
    # add extra dimensions to add the padding
    # to the attention logits. 
    # this will allow for broadcasting later when comparing sequences
    return seq[:, tf.newaxis, :] 

def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(embedding_dim)  # (batch_size, seq_len, embedding_dim)
    ])
    
# Create the model (encoder + decoder setup)
class MultiTaskModel(tf.keras.Model):
    def __init__(self, encoder_layer, decoder_layer, **kwargs):
        super(MultiTaskModel, self).__init__(**kwargs)  # Pass kwargs to the parent Model class
        self.encoder = encoder_layer
        self.decoder = decoder_layer

    def call(self, inputs, training=False, mask=None):
        
        if mask is None:
            mask = create_padding_mask(tf.reduce_sum(inputs, axis=-1))#, self.encoder.num_heads)  # Create mask based on padding tokens

        encoder_output = self.encoder(inputs, training=training, mask=mask)
        outputs = self.decoder(encoder_output, encoder_output, training=training, mask=mask)
        return outputs
    
    def get_config(self):
        # Return a dictionary of configuration that can be used to reinstantiate the model
        config = super(MultiTaskModel, self).get_config()
        config.update({
            'encoder_layer': self.encoder,
            'decoder_layer': self.decoder
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Use the configuration to recreate the model
        encoder = tf.keras.layers.deserialize(config.pop('encoder_layer'))
        decoder = tf.keras.layers.deserialize(config.pop('decoder_layer'))
        return cls(encoder, decoder, **config)
    
    

class EncoderSingleLayer(tf.keras.layers.Layer):
    """
    The encoder layer is composed by a multi-head self-attention mechanism,
    followed by a simple, positionwise fully connected feed-forward network. 
    This architecture includes a residual connection around each of the two 
    sub-layers, followed by layer normalization.
    """
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6, debugLevel = 0, **kwargs):
        super(EncoderSingleLayer, self).__init__(**kwargs)

        self._debugLevel = debugLevel
        if self._debugLevel > 0: print("EncoderSingleLayer -- Init EncoderSingleLayer")
        
        self.num_heads = num_heads
        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim,
                                      dropout=dropout_rate)

        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)

        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = Dropout(dropout_rate)
        
        global ID_EncoderSingleLayer_g
        
        self.myID = ID_EncoderSingleLayer_g
        ID_EncoderSingleLayer_g += 1
    
    ## Definition of the build method to manage properly saving and loading the model
    #def build(self, input_shape):
    #    # Keras will pass the input shape here, you can initialize weights if needed
    #    super(EncoderSingleLayer, self).build(input_shape)  # Always call the parent build method

    
    def call(self, x, training, mask):
        """
        Forward pass for the Encoder Layer
        
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            mask -- Boolean mask to ensure that the padding is not 
                    treated as part of the input
        Returns:
            encoder_layer_out -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
        """
        if self._debugLevel > 0: print("EncoderSingleLayer -- Call EncoderSingleLayer")
        if self._debugLevel > 0: print("EncoderSingleLayer -- ID EncoderSingleLayer Layer: ", self.myID)
        
        # Generate a mask with the correct shape for multi-head attention
        if mask is not None:
            mask = create_padding_mask(tf.reduce_sum(x, axis=-1))
        
        if self._debugLevel > 0: print("EncoderSingleLayer -- Mask shape = ",mask.shape)
        if self._debugLevel > 0: print("EncoderSingleLayer -- x shape = ",x.shape)
        
        global nbCallEncoderSingleLayer_g
        nbCallEncoderSingleLayer_g += 1
        if self._debugLevel > 0: print("EncoderSingleLayer -- Number of call to EncoderSingleLayer: ", nbCallEncoderSingleLayer_g)
        
        # calculate self-attention using mha(~1 line).
        # Dropout is added by Keras automatically if the dropout parameter is non-zero during training
        self_mha_output = self.mha(x,x,x,mask,return_attention_scores=False,training=training)  # Self attention (batch_size, input_seq_len, embedding_dim)
        
        # skip connection
        # apply layer normalization on sum of the input and the attention output to get the  
        # output of the multi-head attention layer (~1 line)
        skip_x_attention = self.layernorm1(x+self_mha_output)  # (batch_size, input_seq_len, embedding_dim)

        # pass the output of the multi-head attention layer through a ffn (~1 line)
        ffn_output = self.ffn(skip_x_attention)  # (batch_size, input_seq_len, embedding_dim)
        
        # apply dropout layer to ffn output during training (~1 line)
        # use `training=training` 
        ffn_output = self.dropout_ffn(ffn_output,training=training)
        
        # apply layer normalization on sum of the output from multi-head attention (skip connection) and ffn output to get the
        # output of the encoder layer (~1 line)
        encoder_layer_out = self.layernorm2(skip_x_attention+ffn_output)  # (batch_size, input_seq_len, embedding_dim)
        
        return encoder_layer_out
        
# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION
class EncoderLayer(tf.keras.layers.Layer):
   """
   The entire Encoder starts by passing the input to an embedding layer 
   and using positional encoding to then pass the output through a stack of
   encoder Layers
       
   """  
   def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim,
              maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6, debugLevel = 0, **kwargs):
       super(EncoderLayer, self).__init__(**kwargs)

       self._debugLevel = debugLevel
       self.embedding_dim = embedding_dim
       self.num_layers = num_layers
       self.num_heads = num_heads
       
       if self._debugLevel > 0: print("EncoderLayer -- Init Encoder Layer")
       if self._debugLevel > 0: print("EncoderLayer -- Num layers: ",num_layers)

       #self.embedding = Embedding(input_vocab_size, self.embedding_dim) -- we will use our own GloVe embedding and pass formated input to the encoder
       self.pos_encoding = BOM.positional_encoding(maximum_position_encoding, 
                                               self.embedding_dim)


       self.enc_layers = [EncoderSingleLayer(embedding_dim=self.embedding_dim,
                                       num_heads=num_heads,
                                       fully_connected_dim=fully_connected_dim,
                                       dropout_rate=dropout_rate,
                                       layernorm_eps=layernorm_eps) 
                          for _ in range(self.num_layers)]

       self.dropout = Dropout(dropout_rate)
       
       global ID_EncoderLayer_g
        
       self.myID = ID_EncoderLayer_g
       ID_EncoderLayer_g += 1
   
   ## Definition of the build method to manage properly saving and loading the model
   #def build(self, input_shape):
   #     # Initialize positional encoding and other weights based on the input shape
   #     super(EncoderLayer, self).build(input_shape)
   #     self.pos_encoding = BOM.positional_encoding(input_shape[1], self.embedding_dim)
        
   def call(self, x, training, mask):
       """
       Forward pass for the Encoder
       
       Arguments:
           x -- Tensor of shape (batch_size, input_seq_len)
           training -- Boolean, set to true to activate
                       the training mode for dropout layers
           mask -- Boolean mask to ensure that the padding is not 
                   treated as part of the input
       Returns:
           x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
       """
       if self._debugLevel > 0: print("EncoderLayer -- Call EncoderLayer")
       if self._debugLevel > 0: print("EncoderLayer -- ID Encoder Layer: ", self.myID)
       if self._debugLevel > 0: print("EncoderLayer -- mask.shape = ",mask.shape)
       if self._debugLevel > 0: print("EncoderLayer -- self.num_layers = ",self.num_layers)
       seq_len = tf.shape(x)[1]
       
       # START CODE HERE
       # Pass input through the Embedding layer
       #x = self.embedding(x)  # (batch_size, input_seq_len, embedding_dim) -- we manage it directly before calling the encoder
       # --> x is preprocessed with our own GloVe embedding
       # Scale embedding by multiplying it by the square root of the embedding dimension
       x *= np.sqrt(self.embedding_dim)
       # Add the position encoding to embedding
       x += self.pos_encoding[:, :seq_len, :]
       # Pass the encoded embedding through a dropout layer
       # use `training=training`
       x = self.dropout(x,training=training)
       # Pass the output through the stack of encoding layers 
       for i in range(self.num_layers):
           x = self.enc_layers[i](x,training=training,mask=mask)
       # END CODE HERE

       return x  # (batch_size, input_seq_len, embedding_dim)

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim, num_classes_question, num_classes_subject, dropout_rate=0.1, layernorm_eps=1e-6, debugLevel = 0, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self._debugLevel = debugLevel
        self.num_heads = num_heads
        
        # Multi-head self-attention for the decoder
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim, dropout=dropout_rate)

        # Fully connected feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(fully_connected_dim, activation='relu'),
            Dense(embedding_dim)
        ])

        # Layer normalization layers
        self.layernorm1 = LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layernorm_eps)

        # Dropout
        self.dropout_ffn = Dropout(dropout_rate)

        # Output heads for different classification tasks
        self.question_head = Dense(num_classes_question, activation='softmax')  # Binary or multi-class
        #self.subject_head = Dense(num_classes_subject, activation='softmax')  # Multi-class for subjects
        #self.action_verb_head = Dense(embedding_dim, activation='softmax')  # Sequence labeling for verbs (softmax over vocab)
        self.objective_head = Dense(2, activation='softmax')  # Binary classification (objective or not)

    ## Definition of the build method to manage properly saving and loading the model
    #def build(self, input_shape):
    #    super(DecoderLayer, self).build(input_shape)
        
    def call(self, x, encoder_output, training, mask):
        """
        Arguments:
            x -- Tensor of shape (batch_size, input_seq_len, embedding_dim)
            encoder_output -- Tensor of shape (batch_size, input_seq_len, embedding_dim) from the encoder
            training -- Boolean, set to true during training
            mask -- Mask for padding
        """
        if mask is not None:
            mask = create_padding_mask(tf.reduce_sum(x, axis=-1))
            
        # Self-attention layer
        attn_output = self.mha(x, x, x, attention_mask=mask, training=training)  # (batch_size, input_seq_len, embedding_dim)
        out1 = self.layernorm1(x + attn_output)  # Residual connection + layer norm

        # Feed-forward layer
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embedding_dim)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection + layer norm

        # Multi-head outputs (classification tasks)
        # Global max pooling or average pooling can be used before the dense layers for classification
        pooled_output = tf.reduce_mean(out2, axis=1)  # Global pooling to get sentence-level embedding (batch_size, embedding_dim)

        # Different tasks (multi-task learning)
        question_output = self.question_head(pooled_output)  # (batch_size, num_classes_question)
        #subject_output = self.subject_head(pooled_output)    # (batch_size, num_classes_subject)
        #action_verb_output = self.action_verb_head(out2)     # (batch_size, input_seq_len, vocab_size or embedding_dim for verbs)
        objective_output = self.objective_head(pooled_output)  # (batch_size, 2 for binary classification)

        return {
            "question": question_output,
            #"subject": subject_output,
            #"action_verb": action_verb_output,
            "objective": objective_output
        }





##############################################################################################
#
# ConversationalEngine : Engine that manages messages from users and prepare answers or follow-up questions
#
#   Main attributes:
#      - BOM                : our Business Object Manager that contains all input data and in which
#                              we will store the output
#
###############################################################################################
class ConversationalEngine:
    """ Generic class """
    def __init__(self, BOM, engine_mode = BOM.TRAINING_MODE_g, filename_transformerModelWrite_p = 'transformer.model.keras', filename_transformerModelRead_p = 'transformer.model.keras', debugLevel = 0):
        """ Constructor """
        self._BOM            = BOM
        self._debugLevel    = debugLevel
        # training or testing
        self._engineMode = engine_mode     # can be BOM.TRAINING_MODE_g or BOM.TEST_MODE_g

        # Define the loss functions for each task
        self.losses = {
            "question": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            #"subject": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            #"action_verb": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            "objective": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        }
        # Define the metrics for each task (optional but useful for monitoring)
        self.metrics = {
            "question": tf.keras.metrics.SparseCategoricalAccuracy(),
            #"subject": tf.keras.metrics.SparseCategoricalAccuracy(),
            #"action_verb": tf.keras.metrics.CategoricalAccuracy(),
            "objective": tf.keras.metrics.SparseCategoricalAccuracy()
        }
        
        # File in which we can store weights
        self.transformer_model_write  = filename_transformerModelWrite_p
        self.transformer_model_read   = filename_transformerModelRead_p
        
        # NLP module
        # Load English model
        self.nlp = spacy.load("en_core_web_sm")
        

    def __str__(self):
        return
    
    """ ========= GETTERS and SETTERS =========== """
    """ Business Objects Manager """
    def _get_BOM(self):
        return self._BOM
    def _set_BOM(self, BOM_l):
        self._BOM = BOM_l
    BOM = property(_get_BOM, _set_BOM)
            
    """ ========= CLASS FUNCTIONS =========== """
    def init_tokenizer(self):
        # Initialize the tokenizer
        self._BOM.tokenizer = Tokenizer(oov_token=self._BOM.oov_token, filters=self._BOM.filters_tokenizer)  # We will set the word index manually
        
        # Create the word index based on our custom vocabulary
        # Note: index 0 is reserved for padding, so indexing starts from 1
        self._BOM.tokenizer.word_index = {word: index for index, word in enumerate(self._BOM.word_to_vec_map.keys(), start=1)}
        # Add the OOV token to the word index
        self._BOM.tokenizer.word_index[self._BOM.tokenizer.oov_token] = len(self._BOM.tokenizer.word_index) + 1
        
        # Manually update the reverse mapping: index_word
        self._BOM.tokenizer.index_word = {index: word for word, index in self._BOM.tokenizer.word_index.items()}
        
        return 0
    
    def build_padded_sequences_from_training_set(self):
        # Preprocess each sentence to handle punctuation properly
        preprocessed_sentences = [BOM.preprocess_text(sentence) for sentence in self._BOM.training_sentences]
        
        # Step 2: Convert sentences to sequences of integers (tokenization)
        sequences = BOM.encode(self._BOM.tokenizer, preprocessed_sentences)
        
        # Preprocess each sentence to keep all the text even if sentences are longer than nbWordsPerSentence
        preprocessed_sequences_withMaxLength_l = []
        idxInTrainingSet_l = 0
        nb_sentences_added_l = 0
        for seq_l in sequences:
            subSequences_l = BOM.generate_subsequences(seq_l,self._BOM.nbWordsPerSentence)
            nb_sentences_added_l = len(subSequences_l)
            preprocessed_sequences_withMaxLength_l.extend(subSequences_l)
            while nb_sentences_added_l > 0:
                self._BOM.indicesInTrainingSet.append(idxInTrainingSet_l)
                nb_sentences_added_l -= 1
            idxInTrainingSet_l += 1
        
        # Step 3: Apply padding to ensure all sequences have the same length
        self._BOM.padded_sequences_training_set = pad_sequences(preprocessed_sequences_withMaxLength_l, maxlen=self._BOM.nbWordsPerSentence, padding='post')
        # DEBUG TRACE
        #for seq_l in self._BOM.padded_sequences_training_set:
        #    print("padded sequence : ", seq_l)
        
        # Step 4: convert output into a tensor
        self._BOM.tensor_sequences_training_set = tf.convert_to_tensor(self._BOM.padded_sequences_training_set, dtype=tf.float32)   
        return 0
        
    def build_padded_sequences(self, list_of_sentences_p):
        # Preprocess each sentence to handle punctuation properly
        preprocessed_sentences = [BOM.preprocess_text(sentence) for sentence in list_of_sentences_p]
        
        # Step 2: Convert sentences to sequences of integers (tokenization)
        sequences = BOM.encode(self._BOM.tokenizer, preprocessed_sentences)
        
        # Preprocess each sentence to keep all the text even if sentences are longer than nbWordsPerSentence
        preprocessed_sequences_withMaxLength_l = []
        
        # DEBUG TRACE
        #print("Test sentences")
        for seq_l in sequences:
            #print(seq_l)
            subSequences_l = BOM.generate_subsequences(seq_l,self._BOM.nbWordsPerSentence)
            preprocessed_sequences_withMaxLength_l.extend(subSequences_l)
            
        # Step 3: Apply padding to ensure all sequences have the same length
        padded_sequences = pad_sequences(preprocessed_sequences_withMaxLength_l, maxlen=self._BOM.nbWordsPerSentence, padding='post')
        # DEBUG TRACE
        #for seq_l in padded_sequences:
        #    print("padded sequence : ", seq_l)
        
        # Step 4: convert output into a tensor
        tensor_sequences_training_set = tf.convert_to_tensor(padded_sequences, dtype=tf.float32)  
        return tensor_sequences_training_set
    
    def preprocess_set_of_sentences(self, list_of_sentences_p):
        # Create a padded sequence
        tensor_sequences_training_set = self.build_padded_sequences(list_of_sentences_p)
        # Apply embeddings
        tensor_embeddings = BOM.map_indices_to_embeddings(tensor_sequences_training_set, self._BOM.tokenizer.index_word, self._BOM.word_to_vec_map)
        
        # DEBUG TRACE
        #print("self embeddings test")
        #for example_l in tensor_embeddings:
        #    print(example_l)
        
        return tensor_embeddings
    
    # Not used, can be deleted if confirmed
    def build_positional_encoding(self):
        # Extract number of dimensions based on word_to_vec structure
        embedding_dim_l = len(next(iter(self._BOM.word_to_vec_map.values())))
        #print("nbEmbeddingPositions :",embedding_dim_l)
        pos_encoding_l = BOM.positional_encoding(self._BOM.nbWordsPerSentence, embedding_dim_l)
        return 0
    
    def prepare_X_and_Y_from_training_set(self):
        # Build the tensor of embeddings from training set
        self._BOM.X_training = BOM.map_indices_to_embeddings(self._BOM.tensor_sequences_training_set, self._BOM.tokenizer.index_word, self._BOM.word_to_vec_map)
        
        # DEBUG TRACE
        #print("self embeddings training")
        #for example_l in self._BOM.X_training:
        #    print(example_l)
            
        # Prepare the structures that will contain the expected answers from training set
        # taking into account that we possibly split the sentences in our preprocess
        questions_reshaped_l = []
        subjects_reshaped_l = []
        action_verbs_reshaped_l = []
        objectives_reshaped_l = []
        for idx_l in self._BOM.indicesInTrainingSet:
            questions_reshaped_l.append(self._BOM.questions[idx_l])
            subjects_reshaped_l.append(self._BOM.subjects[idx_l])
            action_verbs_reshaped_l.append(self._BOM.action_verbs[idx_l])
            objectives_reshaped_l.append(self._BOM.objectives[idx_l])
        
        # Build the tensor that contains the categorizations of the sentences from training set
        # Remark : for now we use only 2 categorizations
        self._BOM.Y_training = {
            "question": tf.convert_to_tensor(questions_reshaped_l, dtype=tf.int32),    # Binary label for question
            "objective": tf.convert_to_tensor(objectives_reshaped_l, dtype=tf.int32)   # Binary label for objective
        }
        return 0
    
    # Sequence of functions to initialize everything after reading word embeddings and training set
    def manage_inputs_and_prepare_structures(self):
        
        self.init_tokenizer()
        #self.build_positional_encoding()
        if self._engineMode == BOM.TRAINING_MODE_g:
            self.build_padded_sequences_from_training_set()
            self.prepare_X_and_Y_from_training_set()
        return
    
    def buildAndCompileTransformerModel(self):
        # Create the model
        if self._debugLevel > 0: print("ConversationalEngine -- Define Encoder Layer")
        encoder_layer_l = EncoderLayer(num_layers=4, embedding_dim=50, num_heads=4, fully_connected_dim=2048,
                    maximum_position_encoding=self.BOM.nbWordsPerSentence, dropout_rate=0.1, layernorm_eps=1e-6)
        if self._debugLevel > 0: print("ConversationalEngine -- Define Decoder Layer")
        decoder_layer_l = DecoderLayer(embedding_dim=50, num_heads=4, fully_connected_dim=2048,
                            num_classes_question=2, num_classes_subject=10)
        if self._debugLevel > 0: print("ConversationalEngine -- Define MultiTask Model")
        self.BOM.model = MultiTaskModel(encoder_layer_l, decoder_layer_l)
        
        # Compile the model
        if self._debugLevel > 0: print("ConversationalEngine -- Compile Model")
        self.BOM.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),      # before : learning_rate=1e-5
                    loss=self.losses,
                    metrics=self.metrics,
                    loss_weights={'question': 2.0, 'objective': 2.0})
        return
    
    def train(self):
        # Train the model
        if(self._debugLevel > 0): print("ConversationalEngine -- Fit Model")
        history = self.BOM.model.fit(self._BOM.X_training, self._BOM.Y_training, epochs=5, batch_size=32, validation_split=0.1)
        
        # Access the accuracy metrics
        history_dict = history.history
        
        # Print all available metrics
        if(self._debugLevel > 0): print("ConversationalEngine -- ", history_dict.keys())
        
        ## Custom Training loop in case we want to monitor closely
        #epochs = 10
        #for epoch in range(epochs):
        #    print(f"Epoch {epoch + 1}/{epochs}")
        #
        #    # Iterate over batches of data
        #    for step, (x_batch, y_batch) in enumerate(train_dataset):
        #        with tf.GradientTape() as tape:
        #            # Forward pass
        #            outputs = model(x_batch, training=True)
        #            
        #            # Compute losses
        #            loss_values = {
        #                task: losses[task](y_batch[task], outputs[task]) for task in outputs.keys()
        #            }
        #            total_loss = sum(loss_values.values())
        #
        #        # Compute gradients and update weights
        #        grads = tape.gradient(total_loss, model.trainable_weights)
        #        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        #
        #        # Optionally track metrics
        #        for task in outputs.keys():
        #            metrics[task].update_state(y_batch[task], outputs[task])
        #
        #    # Display progress
        #    for task in outputs.keys():
        #        print(f"{task} Loss: {loss_values[task].numpy()}, {task} Accuracy: {metrics[task].result().numpy()}")
        #        metrics[task].reset_states()
    
    def saveModel(self):
        # Save the weights after training
        self.BOM.model.save(self.transformer_model_write)
        print("Model saved !")
        return
    
    def loadModel(self):
        # Load the weights from a previous training
        #self.BOM.model.load_weights(self.transformer_weights_read)
        self.BOM.model = tf.keras.models.load_model(self.transformer_model_read,custom_objects={'MultiTaskModel': MultiTaskModel})
        print("Model loaded !")
    
    def predict(self, X_test):
        # call the model to predict the classifications on a set of sentences X_test
        predictions = self.BOM.model.predict(X_test)
        return predictions
        
    def extract_objective(self,sentence):
        # Parse the sentence using spaCy
        doc = self.nlp(sentence)
        
        # Look for the word "goal", "objective", or similar, and track the verb following it
        for token in doc:
            if token.text.lower() in ["goal", "objective", "target", "plan"]:
                # Once we find "goal", we need to move forward to find the verb and its complements
                for next_token in token.head.children:
                    if next_token.dep_ in ["attr", "xcomp", "ccomp", "acl"]:
                        # Extract the full verb phrase subtree following the verb
                        objective_phrase = " ".join([t.text for t in next_token.subtree])
                        return objective_phrase
                    # Look for the verb immediately after "goal" or "objective"
                    if next_token.pos_ == "VERB":
                        for child in next_token.children:
                            if child.dep_ in ["xcomp", "ccomp"]:
                                # Extract the full verb phrase subtree
                                objective_phrase = " ".join([t.text for t in child.subtree])
                                return objective_phrase
        return "No clear objective found"
    
    ## Test the function
    #sentence = "My goal is to be able to run a marathon by the end of 2024."
    #objective = extract_objective(sentence)
    #print(f"Extracted Objective: {objective}")
    
    def extract_action(self,sentence):
        # Parse the sentence using spaCy
        doc = self.nlp(sentence)
        
        actions = []
        
        # Find verbs and their related noun phrases or objects
        for token in doc:
            # Look for main verbs
            if token.pos_ == "VERB" and token.text.lower() not in ["is", "want", "would", "like","will"]:
                action = token.text
                
                # Start by capturing the subtree of the verb
                action_phrase = [token.text]
                
                # Look for direct objects (dobj), prepositional phrases (prep), or complements (xcomp, ccomp)
                for child in token.children:
                    if child.dep_ in ["dobj", "prep", "xcomp", "ccomp", "attr", "acomp", "pobj"]:
                        # Append the child subtree to maintain order of tokens
                        action_phrase += [t.text for t in child.subtree]
    
                # Join the action phrase to form a coherent action
                full_action = " ".join(action_phrase)
                actions.append(full_action)
                break
                
                # V1
                ## Look for direct objects (dobj), prepositional phrases (prep), or complements (xcomp, ccomp)
                #for child in token.children:
                #    if child.dep_ in ["dobj", "prep", "xcomp", "ccomp", "attr", "acomp", "pobj"]:
                #        print("child: ",child.text.lower())
                #        print("child dep: ",child.dep_)
                #        # Append the action + complement (verb + object or complement)
                #        actions.append(f"{action} {child.text}")
                #        # Add the subtree of the child to capture full phrases (e.g., "run a marathon")
                #        action_phrase = " ".join([t.text for t in child.subtree])
                #        actions.append(f"{action} {action_phrase}")
    
        # If no action found, return a message
        if not actions:
            return ""
        
        # Return the most meaningful action phrase
        return actions
    
    ## Test with sample sentences
    #sentence1 = "My goal is to be able to run a marathon by the end of 2024."
    #sentence2 = "I want to cook dinner for my girlfriend."
    #
    #action1 = extract_action(sentence1)
    #action2 = extract_action(sentence2)
    #
    #print(sentence1)
    #print(f"Extracted Action (Sentence 1): {action1}")
    #print(sentence2)
    #print(f"Extracted Action (Sentence 2): {action2}")
    
    def extract_temporality(self,sentence):
        # Parse the sentence using spaCy
        doc = self.nlp(sentence)
        
        temporality = []
        
        # 1. Check for temporal entities like DATE, TIME, and DURATION
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME", "DURATION"]:
                temporality.append(ent.text)
    
        # 2. Check for temporal prepositional phrases using dependency parsing
        for token in doc:
            if token.dep_ == "prep" and token.text.lower() in ["by", "before", "in", "until", "during", "after"]:
                # V1
                ## Collect the full prepositional phrase indicating temporality
                #temp_phrase = " ".join([t.text for t in token.subtree])
                #temporality.append(temp_phrase)
                # Start by capturing the subtree of the verb
                action_phrase = [token.text]
                
                # Look for direct objects (dobj), prepositional phrases (prep), or complements (xcomp, ccomp)
                for child in token.children:
                    if child.dep_ in ["dobj", "prep", "xcomp", "ccomp", "attr", "acomp", "pobj"]:
                        # Append the child subtree to maintain order of tokens
                        action_phrase += [t.text for t in child.subtree]
    
        if not temporality:
            return "No temporality found"
        
        return " ".join(action_phrase)
    
    ## Test the function
    #sentence = "My goal is to be able to run a marathon by the end of 2024."
    #temporal_info = extract_temporality(sentence)
    #print(sentence)
    #print(f"Extracted Temporality: {temporal_info}")
    #
    ## Another test
    #sentence2 = "I plan to finish the project before next summer."
    #temporal_info2 = extract_temporality(sentence2)
    #print(sentence2)
    #print(f"Extracted Temporality: {temporal_info2}")
        

