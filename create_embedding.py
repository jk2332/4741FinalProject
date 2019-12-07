import sentencepiece as spm
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
from absl import logging
import numpy as np
import csv
import math
import pandas as pd



module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))

with tf.Session() as sess:
    spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
print("SentencePiece model loaded at {}.".format(spm_path))


def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape=(len(ids), max_len)
    values=[item for sublist in ids for item in sublist]
    indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)



df = pd.read_csv('listings.csv')
summaries = df['summary'].fillna('')
messages = summaries

values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)

# Reduce logging output.
logging.set_verbosity(logging.ERROR)

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    message_embeddings = session.run(
        encodings,
        feed_dict={input_placeholder.values: values,
                   input_placeholder.indices: indices,
                   input_placeholder.dense_shape: dense_shape})
    #
    #
    # for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    #     print("Message: {}".format(messages[i]))
    #     print("Embedding size: {}".format(len(message_embedding)))
    #     message_embedding_snippet = ", ".join(
    #         (str(x) for x in message_embedding[:3]))
    #     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

with open('summary_embeddings.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(message_embeddings.tolist())