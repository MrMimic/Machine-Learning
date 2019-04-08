#!/usr/bin/env python3


# How to build a semantic SE in 10 lines of code
# ----------------------------------
# DOC:
# https://github.com/makcedward/nlp/blob/master/sample/nlp-embeddings-sentence-elmo.ipynb
# https://gist.github.com/ranarag/77014b952a649dbaf8f47969affdd3bc
# ----------------------------------


import re
import sys
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing.sequence import pad_sequences as PS


""" OPTIONS """
TRAINING_FILE = '0_tester.txt'
CLUSTER_NODES = 20


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


if '--train' in sys.argv:

        # Get last model version
        elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=True)
                       
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

        # Save vectors
        with open('elmo_vectors.pickle', 'wb') as handler:
                pickle.dump(vectors, handler, protocol=pickle.HIGHEST_PROTOCOL)
        
         
if '--query' in sys.argv:
        
        # Check len of provided arguments
        if len(sys.argv) > 3:
                print('Please provided query with brackets (eg. python3 main.py --query "Hello john, how you doing?"')
                exit(0)
        else:
                QUERY = sys.argv[2]

        # Load trained vectors
        with open('elmo_vectors.pickle', 'rb') as handler:
                vectors = pickle.load(handler)

        # Get len of a vector to pad the question
        sentences, original_tokens, max_length = open_file_and_pad_sequences(TRAINING_FILE)
        
        # Get last model version
        elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False)

        # Vectorize query
        tokens_query = re.findall('\w+', QUERY)
        padded_query = PS(sequences=[tokens_query], dtype=object, maxlen=max_length, padding='post', value='<PAD>')[0]
        embedded_query =  elmo(padded_query, signature="default", as_dict=True)['elmo']

        # Start Tensorflow
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(inter_op_parallelism_threads=CLUSTER_NODES, intra_op_parallelism_threads=CLUSTER_NODES)

        # Run the graph        
        with tf.Session(config=config) as sess:
                sess.run(init_op)
                vectorized_query = sess.run(embedded_query)[0]
               
                # Calculate cosines query VS all vectors            
                cosines = list()
                for vector, text in zip(vectors, original_tokens):
                        global_sim = cosine_sim(vector, vectorized_query)
                        cosines.append(sess.run(1-tf.reduce_mean(global_sim)))

        # Now, sort lists, first elements are closer to the query
        vectors = list(vectors)
        sentences = [' '.join(tokens) for tokens in original_tokens]
        sorted_results = sorted(zip(cosines, sentences, vectors), key=lambda x: x[0], reverse=True)
        
        # Printout results
        for position, result in enumerate(sorted_results):
                print('#{}\t{}\t{}'.format(position+1, result[0], result[1]))
