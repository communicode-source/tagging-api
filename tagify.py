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
        pass

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

    def train(self, data):
        """
        Trains the model on a dataset
        :param data: List of form [[X1, Y1], ..., [XN, YN]]
        :return: Void
        """
        pass

    def save(self, fname):
        """
        Save a model for future use
        :param fname: Filename without extension
        :return: Void
        """
        pass

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

def classToVector(targets, classes):
    """
    Converts a list of classes into a hot vector over all classes
    :param targets: Set of classes to encode
    :param classes: Set of total classes as template
    :return: List of 1s and Os
    """
    return [1 if c in targets else 0 for c in classes]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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