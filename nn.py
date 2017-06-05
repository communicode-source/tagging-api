# nn.py
# Communicode-tagify
# Author : Abraham Oliver <abeoliver.116@gmail.com>
# Date : 6/4/2017

import tensorflow as tf
import tagify
from pickle import dump, load
import numpy as np

class NeuralNetwork (tagify.Model):
    def __init__(self, stems, classes, layers, ignore = []):
        """
        Initialize
        :param stems: List of Strings
        :param classes: List of Strings
        :param layers: List of Integers
        :param ignore: List of Strings (defualt [])
        """
        self.stems = stems
        self.classes = classes
        self.layers = layers
        self.ignore = ignore
        # Initialize model
        self.w = None
        self.b = None

    def vectorize(self, phrase):
        """
        Converts a phrase into a vector using a given bag of words
        :param phrase: String to be vectorized
        :return: List of 1s and 0s
        """
        # Stem and tokenize new phrase
        # Note: ignore is not needed because stem list
        # has already ingored tokens
        tas = tagify.tokenizeAndStem(phrase, ignore = self.ignore)
        # Make zero array
        array = [1.0 if i in tas else 0.0 for i in self.stems]
        return array

    def classify(self, vector):
        """
        Converts a class vector into classes and confidence values
        :param vector: List of confidence values
        :return: Dictionary of Class -> Confidence
        """
        return {self.classes[i]: vector[0][i] for i in range(len(vector[0]))}

    def feed(self, vector):
        """
        Feeds a vector through the model
        :param vector: Vector representing a phrase (produced by vectorize)
        :return: Vector representing class confidences
        """
        # Numpy version of calc
        def _calc(inp, w, b, n=0):
            """Recursive function for feeding through layers"""
            # End recursion
            if n == len(self.layers) - 2:
                # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
                return np.dot(inp, w[n]) + b[n]
            # Continue recursion
            return _calc(np.dot(inp, w[n]) + b[n], w, b, n + 1)
        return tagify.sigmoid(_calc([vector], self.w, self.b))

    def train(self, trainX, trainY, epochs, learnRate):
        """
        Trains a model
        :param trainX: Training Inputs
        :param trainY: Training labels
        :param epochs: Number of epochs to run the training
        :param learnRate: Learning rate for the optimizer
        :return: Void
        """
        # Create Tensorflow session
        session = tf.Session()

        # Tensorflow version of calculate
        def _calc(inp, w, b, n=0):
            """Recursive function for feeding through layers"""
            # End recursion
            if n == len(self.layers) - 2:
                # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
                return tf.matmul(inp, w[n], name="mul{0}".format(n)) + b[n]
            # Continue recursion
            return _calc(tf.matmul(inp, w[n], name="mul{0}".format(n)) + [n], w, b, n + 1)

        # Define model
        # Input
        x = tf.placeholder(tf.float32, [None, self.layers[0]], name = "x")
        # Weights
        w = [tf.Variable(tf.random_normal([self.layers[n], self.layers[n + 1]]), name = "w{0}".format(n))
             for n in range(len(self.layers) - 1)]
        # Biases (1 X Ouput Size)
        b = [tf.Variable(tf.random_normal([1, self.layers[n + 1]]), name = "b{0}".format(n))
             for n in range(len(self.layers) - 1)]
        # Output
        y = _calc(x, w, b)
        # Labels
        y_ = tf.placeholder(tf.float32, [None, self.layers[-1]], name = "y_")
        # Loss function
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = y, name = "loss")
        # loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        # Optimizer
        trainStep = tf.train.ProximalGradientDescentOptimizer(learnRate).minimize(loss)
        # Initialize variables
        session.run(tf.global_variables_initializer())

        # Training
        with session.as_default():
            # Run train step "epochs" times
            for i in range(epochs):
                # Run train step
                session.run(trainStep, feed_dict={x: trainX, y_: trainY})
            # Save parameters
            self.w = [i.eval() for i in w]
            self.b = [i.eval() for i in b]

    def save(self, fname):
        """
        Saves a model
        :param fname: String for filename
        :return: Void
        """
        # Open a binary file to write the model to
        with open("{0}.model".format(fname), "wb") as f:
            dump(self, f)

def getModel(fname):
    """
    Gets a saved model from storage
    :param fname: String file name
    :return: NeuralNetwork
    """
    # Open a binary file to get the model from
    with open("{0}.model".format(fname), "rb") as f:
        model = load(f)
    return model