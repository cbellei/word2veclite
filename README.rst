.. highlight:: rst

^^^^^^^^^^^^
Word2VecLite
^^^^^^^^^^^^
Word2VecLite is a Python implementation of Word2Vec that makes it easy to understand how Word2Vec works.
This package is intended to be used in conjunction with `this blog post <http://www.claudiobellei.com/2018/01/07/backprop-word2vec-python/>`__.

Installation
============
* In your target folder, clone the repository with the command::

        git clone https://github.com/cbellei/word2veclite.git

* Then, inside the same folder (as always, it is advisable to use a virtual environment)::

        pip install .

NOTE: if you get a `ModuleNotFoundError: No module named 'setuptools.wheel'` error, you may need to update
pip and setuptools via the command `pip install -U pip setuptools`

* Make sure you install the correct dependencies by typing `pip install -r requirements.txt`

NOTE: this should work on both a conda environment and a standard virtual environment

* To check that the package has been installed, in the Python shell type::

        import word2veclite

* If everything works correctly, the package will be imported without errors.

Dependencies
============
* Word2VecLite is tested on Python 3.6 and depends on NumPy, Keras (see ``requirements.txt`` for version
information). The unit tests in test/test_word2veclite.py depend on Tensorflow.

How to use Word2VecLite
=======================
**Input**. You need to:
1. Define a corpus of text (the vocabulary will be built from it)
2. Define which method you want to use: `cbow` or `skipgram`
3. Decide how many nodes should make the hidden layer
4. Define the size of the context window around the center word
5. Define learning rate
6. Decide how many epochs you want to train the neural network for

**Output**. Word2VecLite outputs the embeddings W1 and W2 of the neural network and the history of the loss vs. epochs.

Example
=======
* In IPython, type::

    from word2veclite import Word2Vec

    corpus = "I like playing football with my friends"
    cbow = Word2Vec(method="cbow", corpus=corpus,
                    window_size=1, n_hidden=2,
                    n_epochs=10, learning_rate=0.8)
    W1, W2, loss_vs_epoch = cbow.run()

    print(W1)
    #[[ 0.99870389  0.20697257]
    # [-1.01911559  2.26364436]
    # [-0.69737232  0.14131477]
    # [ 3.28315183  1.13801973]
    # [-1.42944927 -0.62142097]
    # [ 0.65359329 -2.21415048]
    # [-0.22343751 -1.17927987]]

    print(W2)
    #[[-0.97080793  1.21120331  2.15603796 -1.79083151  3.38445043 -1.65295511
    #   1.36685097]
    # [2.77323464  0.78710269  2.74152617  0.08953005  0.04400675 -1.34149651
    #   -2.19375528]]

    print(loss_vs_epoch)
    #[14.328868654443703, 12.290456644464603, 10.366644621637064,
    # 9.1759777684446622, 8.4233626997233895, 7.3952948684910256,
    # 6.1727393307549736, 5.1639476117698191, 4.6333377088153043,
    # 4.2944697259465485]

Licence
=======
`Apache License, Version
2.0 <https://github.com/cbellei/abyes/blob/master/LICENSE>`__
