from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import numpy as np


def tokenize(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus_tokenized = tokenizer.texts_to_sequences(corpus)
    V = len(tokenizer.word_index)
    return corpus_tokenized, V


def initialize(V, N):
    np.random.seed(100)
    W1 = np.random.rand(V, N)
    W2 = np.random.rand(N,V)

    return W1, W2


def corpus2io(corpus_tokenized, V, window_size):
    """Converts corpus text into context and label arrays
    # Arguments
        corpus_tokenized: corpus text
        window_size: size of context window
    # Returns
        context and label arrays
    """
    for words in corpus_tokenized:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            labels = []
            s = index - window_size
            e = index + window_size + 1
            contexts.append([words[i]-1 for i in range(s, e) if 0 <= i < L and i != index])
            labels.append(word-1)
            x = np_utils.to_categorical(contexts, V)
            y = np_utils.to_categorical(labels, V)
            yield (x, y.ravel())


def softmax(x):
    """Calculate softmax based probability for given input vector
    # Arguments
        x: numpy array/list
    # Returns
        softmax of input array
    """
    return np.exp(x) / np.sum(np.exp(x))
