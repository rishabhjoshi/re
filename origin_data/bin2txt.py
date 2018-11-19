
# Author : Rishabh Joshi
# Insti  : IISc Bangalore

from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format('vec.bin', binary = True)
model.save_word2vec_format('vec.txt', binary = False)
