# tagify.py
# Communicode-tagify
# Author : Abraham Oliver <abeoliver.116@gmail.com>
# Date : 5/26/2017

# Import dependencies
import numpy as np
import tensorflow as tf
from importlib import import_module, __import__
# Import natural language tools
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer

# Initialize Stmmer for token stemming
stemmer = LancasterStemmer()

class Model (object):
    """
    Holds a `model` that predicts tags
    """
    def __init__(self, w, b, stems, classes):
        """
        Initializes a model
        :param w: Array of weights
        :param b: Array of biases
        :param stems: Stem list
        :param classes: Dictionary of classes to names
        """
        self.w = w
        self.b = b
        self.stems = stems
        self.classes = classes

    def tag(self, phrase):
        """
        Tags a phrase
        :param phrase: String
        :return: List of Strings
        """
        p = self.vectorize(phrase)
        return self.classify(self.feed(p))

    def feed(self, vector):
        """
        Feed a vector to model
        :param vector: A vector representing a phrase in `vectorize` form
        :return: Output vector representing the class confidences
        """
        return vector

    def vectorize(self, phrase):
        """
        Convert a phrase into a vector
        :param phrase: Input string
        :return: Vector representing the phrase
        """
        return np.array([0])

    def classify(self, vector):
        """
        Convert output class vector into classes with confidences
        :param vector: Vector representing class confidences
        :return: Dictionary of class names and confidences
        """
        return {}

def tokenizeAndStem(phrase, ignore = []):
    """
    Tokenizes and stems the words of a phrase and removes unwanted characters
    :param phrase: String to be tokenized
    :param ignore: List of strings to be ignored (default [])
    :return: List of strings
    """
    # Final list of stems
    stems = []
    # Loop through tokenized items
    for word in word_tokenize(phrase):
        # Ignore unimporant character
        if word not in ignore:
            # Stem and lowercase each word
            stems.append(stemmer.stem(word.lower()))
    return stems

def phraseToVectorBOW(phrase, stems):
    """
    Converts a phrase into a vector using a given bag of words
    :param phrase: String to be vectorized
    :param stems: Ordered list of stems as template
    :return: List of 1s and 0s
    """
    # Stem and tokenize new phrase
    # Note: ignore is not needed because stem list
    # has already ingored tokens
    tas = tokenizeAndStem(phrase)
    # Make zero array
    array = [1 if i in tas else 0 for i in stems]
    return array

def classToVector(classIds, classes):
    """
    Converts a list of classes into a hot vector over all classes
    :param classIds: Set of classes to encode
    :param classes: Set of total classes as template
    :return: List of 1s and Os
    """
    return [1 if c in classIds else 0 for c in classes]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def getModel(fBaseName):
    """
    Gets a Neural Network model with a given base file name
    :param fBaseName: String of file name base (ie. without _w or _b)
    :return: Function of Vector -> Vector
    """
    # Import weights
    w = np.loadtxt(
        "C:\\Users\\abeol\\Git\\communicode-tagify\\models\\" + fBaseName + "_w")
    # Import biases
    b = np.loadtxt(
        "C:\\Users\\abeol\\Git\\communicode-tagify\\models\\" + fBaseName + "_b")
    # Import data module
    name = "C:\\Users\\abeol\\Git\\communicode-tagify\\models\\" + fBaseName + ".py"
    data = __import__(name)
    return Model(w, b, data.stems, data.classes)

def tagify(phrase, model):
    """
    Tagifies the given phrase
    :param phrase: String to tagify
    :param model: A Model
    :return: List of Strings
    """
    tagConfidences = model.feed(phrase)
    tags = [{classList[i]: np.around(tagConfidences[i], decimals = 3)}
            for i in range(len(tagConfidences))
            if tagConfidences[i] > 0]
    return tags

def getStems(data, ignore):
    """
    Get unique stems from data
    :param data: list of dictionaries with a "class" and a "phrase"
    :param ignore: List of strings
    :return: List of strings
    """
    # List of unique stems and the number of their occurences
    stems = []
    # Tokenize, stem, and add all occurences to stems
    for d in data:
        # Tokenize and stem the phrase
        t = tokenizeAndStem(d["phrase"], ignore)
        # Protect from duplicates
        for word in t:
            if word in stems: continue
            else: stems.append(word)
    return sorted(stems)

def getDataSet(trainData, classes, stems):
    """
    Creates a dataset
    :param trainData: List of dictionaries with "class" and "phrase" fields
    :param classes: List of classes used in data
    :param stems: List of strings
    :return:
    """
    # Input Vectors
    X = [phraseToVectorBOW(i["phrase"], stems)
         for i in trainData]
    # Labels / Classes / Outputs
    Y = [classToVector(i["class"], classes) for i in trainData]
    # Return pair
    return [X, Y]

def saveModel(fname, w, b):
    """
    Saves a model
    :param fname: Filename
    :param w: Numpy array of weights
    :param b: Numpy array of biases
    :return: Void
    """
    np.savetxt(fname+"_w", w)
    np.savetxt(fname+"_b", b)

def train(trainX, trainY, stems, fname, epochs, learnRate):
    """
    Trains a model
    :param trainX: Training Inputs
    :param trainY: Training labels
    :param stems: List of stems
    :param fanme: File name for model saving
    :param epochs: Number of epochs to run the training
    :param learnRate: Learning rate for the optimizer
    :return: Void
    """
    # Create Tensorflow session
    session = tf.Session()

    # Network layer sizes
    # [Number of stems in bag of words, .... Hidden ..., Number of Classes]
    layers = [len(trainX[0]), len(trainY[1])]

    # Define model
    # Input
    x = tf.placeholder(tf.float32, [None, layers[0]])
    # Weights (Input Size X Output Size)
    w = tf.Variable(tf.random_normal([layers[0], layers[1]]))
    # Biases (1 X Ouput Size)
    b = tf.Variable(tf.random_normal([1, layers[1]]))
    # Output
    y = tf.sigmoid(tf.matmul(x, w) + b)
    # Labels
    y_ = tf.placeholder(tf.float32, [None, layers[1]])
    # Loss function
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y_, logits = y)
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
            session.run(trainStep, feed_dict = {x: trainX, y_: trainY})
    # Save model
    saveModel(fname, w.eval(session = session), b.eval(session = session))