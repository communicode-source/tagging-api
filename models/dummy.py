# dummy.py
# Communicode-tagify
# Author : Abraham Oliver <abeoliver.116@gmail.com>
# Date : 5/29/2017

# Get tagify tools and JSON
import tagify
from nn import NeuralNetwork

# Tokens to ignore
ignore = ["?", "'s", ".", "!", "I", "i", "We", "we", "need",
          "want", "would", "like", "and", "or", "a", "to", "with"]

# Dummy Data
trainData = [
    {"class": ["HTML, CSS"], "phrase": "Website"},
    {"class": ["HTML, CSS"], "phrase": "Webpage"},
    {"class": ["HTML, CSS"], "phrase": "I need a website or a webpage with links and pictures"},
    {"class": ["Database"], "phrase": "Database"},
    {"class": ["Database"], "phrase": "Saves user profiles"},
    {"class": ["Database"], "phrase": "I need a database that saves information about users"},
    {"class": ["Python or Ruby"], "phrase": "API"},
    {"class": ["Python or Ruby"], "phrase": "Application"},
    {"class": ["HTML, CSS", "Python or Ruby"], "phrase": "I need a web application that sends requests"},
    {"class": ["Branding"], "phrase": "Brand and designs"},
    {"class": ["Branding"], "phrase": "Logos and posters"},
    {"class": ["Branding"], "phrase": "I need to re-brand with new designs and logos"},
    {"class": ["HTML, CSS", "Database"], "phrase": "I want users to sign in to a website"},
    {"class": ["HTML, CSS", "Database"], "phrase": "I need a database and webpage that takes emails"}
]

# Classes
classes = ["HTML, CSS", "Database", "Python or Ruby", "Branding"]

# Stems
stems = ['about', 'ap', 'apply', 'brand', 'databas', 'design', 'email', 'in',
         'inform', 'link', 'logo', 'new', 'pict', 'post', 'profil', 're-brand',
         'request', 'sav', 'send', 'sign', 'tak', 'that', 'us', 'web', 'webp',
         'websit']


if __name__ == "__main__":
    # If this file is run independently, train model
    n = NeuralNetwork(stems, classes, [len(stems), 25, 25, len(classes)], ignore)
    # Train
    # Inputs
    data = [(n.vectorize(i["phrase"]),
             tagify.classToVector(i["class"], classes))
            for i in trainData]
    n.train([i[0] for i in data],
            [i[1] for i in data],
            10000,
            .05)
    n.save("dummy2")
    """
    model = getModel("C:\\Users\\abeol\\Git\\communicode-tagify\\models\\dummy")
    model.tag("User database")
    """