#!/usr/bin/env python3


# How to build a semantic SE in 10 lines of code
# ----------------------------------
# DOC:
# https://github.com/makcedward/nlp/blob/master/sample/nlp-embeddings-sentence-elmo.ipynb
# https://gist.github.com/ranarag/77014b952a649dbaf8f47969affdd3bc
# ----------------------------------


import re
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.sequence import pad_sequences as PS

""" OPTIONS """
TRAINING_FILE = '0_tester.txt'
CLUSTER_NODES = 20
QUERY = 'le chat est sur le canapé' 

# Get last model version
elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True)

# Create a padding function
def open_file_and_pad_sequences(file_name):
        """
        This function will open a file and pad each line to the same number of tokens
        """
        with open(file_name, 'r') as handler:
                # Clean and tokenize lines
                lines = handler.readlines()
        lines = [line.strip('\n').lower() for line in lines]
        tokens = [re.findall('\w+', line) for line in lines]
        tokens = [list(map(str, token)) for token in tokens]
        # Calculate max_length        
        max_length = max([len(line) for line in tokens])
        # Now use Keras padding utility cause of lazyness
        padded = PS(sequences=tokens,dtype=object, maxlen=max_length, padding='post', value='<PAD>')
        joined = [' '.join(line) for line in padded]
        del lines, padded
        return joined, tokens, max_length 
                               
# Use that lazy function to create same length sentences
sentences, original_tokens, max_length = open_file_and_pad_sequences(TRAINING_FILE)

# Embed our documents
embeddings = elmo(sentences, signature="default", as_dict=True)['elmo']

# Initialize tensorflow
init_op = tf.global_variables_initializer() 
config = tf.ConfigProto(inter_op_parallelism_threads=CLUSTER_NODES, intra_op_parallelism_threads=CLUSTER_NODES)

# Run the graph
with tf.Session(config=config) as sess:
        sess.run(init_op)
        vectors = sess.run(embeddings)

        # Thoses guys are our model, let's embed a query
        tokens_query = re.findall('\w+', QUERY)
        padded_query = PS(sequences=[tokens_query], dtype=object, maxlen=max_length, padding='post', value='<PAD>')[0]
        embedded_query =  elmo(padded_query, signature="default", as_dict=True)['elmo']
        vectorized_query = sess.run(embedded_query)[0]

# Define a cosine similarity function
def cosine_sim(x1, x2, name='Cosine_loss'):
        with tf.name_scope(name):
                # Define placeholders
                x1_val = tf.sqrt(tf.reduce_sum(tf.matmul(x1,tf.transpose(x1)),axis=1))
                x2_val = tf.sqrt(tf.reduce_sum(tf.matmul(x2,tf.transpose(x2)),axis=1))
                # Get numerator and denominator
                denom =  tf.multiply(x1_val,x2_val)
                num = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
                #Compute
                return tf.math.divide(num,denom)

# Calculate cosine similarity for each vector
with tf.Session() as sess:
        print()
        for vector, text in zip(vectors, original_tokens):
                global_sim = cosine_sim(vector, vectorized_query)
                print(' '.join(text))
                sess.run(tf.print(global_sim))
                sess.run(tf.print(1-tf.reduce_mean(global_sim)))
                print()
                
                
