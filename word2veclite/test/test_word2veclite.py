import unittest
from unittest import TestCase
from word2veclite.word2veclite import Word2Vec
import numpy as np
import tensorflow as tf


class TestFunctions(TestCase):

    learning_rate = 0.1
    W1 = np.ones((7, 2))
    W2 = np.array([[0.2, 0.2, 0.3, 0.4, 0.5, 0.3, 0.2],
                   [0.3, 0., 0.1, 0., 1., 0.1, 0.5]])
    V, N = W1.shape
    context_words = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]])
    center_word = np.array([0., 0., 1., 0., 0., 0., 0.])

    def test_cbow(self):
        cbow = Word2Vec(learning_rate=self.learning_rate)
        W1_m, W2_m, loss_m = cbow.cbow(np.asmatrix(self.context_words), np.asmatrix(self.center_word), self.W1, self.W2, 0.)

        with tf.name_scope("cbow"):
            x = tf.placeholder(shape=[self.V, len(self.context_words)], dtype=tf.float32, name="x")
            W1_tf = tf.Variable(self.W1, dtype=tf.float32)
            W2_tf = tf.Variable(self.W2, dtype=tf.float32)
            hh = [tf.matmul(tf.transpose(W1_tf), tf.reshape(x[:, i], [self.V, 1]))
                  for i in range(len(self.context_words))]
            h = tf.reduce_mean(tf.stack(hh), axis=0)
            u = tf.matmul(tf.transpose(W2_tf), h)
            loss_tf = -u[int(np.where(self.center_word == 1)[0])] + tf.log(tf.reduce_sum(tf.exp(u), axis=0))
            grad_W1, grad_W2 = tf.gradients(loss_tf, [W1_tf, W2_tf])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            W1_tf, W2_tf, loss_tf, dW1_tf, dW2_tf = sess.run([W1_tf, W2_tf, loss_tf, grad_W1, grad_W2],
                                                             feed_dict={x: self.context_words.T})

        W1_tf -= self.learning_rate * dW1_tf
        W2_tf -= self.learning_rate * dW2_tf

        for i in range(self.V):
            for j in range(self.N):
                self.assertAlmostEqual(W1_m[i, j], W1_tf[i, j], places=5)

        for i in range(self.N):
            for j in range(self.V):
                self.assertAlmostEqual(W2_m[i, j], W2_tf[i, j], places=5)

        self.assertAlmostEqual(loss_m, float(loss_tf), places=5)

    def test_skipgram(self):
        skipgram = Word2Vec(learning_rate=self.learning_rate)
        W1_m, W2_m, loss_m = skipgram.skipgram(np.asmatrix(self.context_words), np.asmatrix(self.center_word), self.W1, self.W2, 0.)

        with tf.name_scope("skipgram"):
            x = tf.placeholder(shape=[self.V, 1], dtype=tf.float32, name="x")
            W1_tf = tf.Variable(self.W1, dtype=tf.float32)
            W2_tf = tf.Variable(self.W2, dtype=tf.float32)
            h = tf.matmul(tf.transpose(W1_tf), x)
            u = tf.stack([tf.matmul(tf.transpose(W2_tf), h) for i in range(len(self.context_words))])
            loss_tf = -tf.reduce_sum([u[i][int(np.where(c == 1)[0])]
                                      for i, c in zip(range(len(self.context_words)), self.context_words)], axis=0)\
                      + tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(u), axis=1)), axis=0)

            grad_W1, grad_W2 = tf.gradients(loss_tf, [W1_tf, W2_tf])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            W1_tf, W2_tf, loss_tf, dW1_tf, dW2_tf = sess.run([W1_tf, W2_tf, loss_tf, grad_W1, grad_W2],
                                                             feed_dict={x: self.center_word.reshape(self.V, 1)})

        W1_tf -= self.learning_rate * dW1_tf
        W2_tf -= self.learning_rate * dW2_tf

        for i in range(self.V):
            for j in range(self.N):
                self.assertAlmostEqual(W1_m[i, j], W1_tf[i, j], places=5)

        for i in range(self.N):
            for j in range(self.V):
                self.assertAlmostEqual(W2_m[i, j], W2_tf[i, j], places=5)

        self.assertAlmostEqual(loss_m, float(loss_tf), places=5)


if __name__ == "__main__":
    unittest.main()
