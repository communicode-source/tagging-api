# dummy.py
# Communicode-tagify
# Author : Abraham Oliver <abeoliver.116@gmail.com>
# Date : 5/29/2017

# Get

# Tokens too ignore
ignore = ["?", "'s", ".", "!", "I", "i", "We", "we", "need",
          "want", "would", "like", "and", "or", "a", "to", "with"]

# Classes
classList = {
    0: "HTML, CSS",
    1: "Database",
    2: "Python or Ruby",
    3: "Branding"
}
classes = range(len(classList.keys()))

# Dummy Data
trainData = [
    {"class": [1], "phrase": "Website"},
    {"class": [1], "phrase": "Webpage"},
    {"class": [1], "phrase": "I need a website or a webpage with links and pictures"},
    {"class": [2], "phrase": "Database"},
    {"class": [2], "phrase": "Saves user profiles"},
    {"class": [2], "phrase": "I need a database that saves information about users"},
    {"class": [3], "phrase": "API"},
    {"class": [3], "phrase": "Application"},
    {"class": [1, 3], "phrase": "I need a web application that sends requests"},
    {"class": [4], "phrase": "Brand and designs"},
    {"class": [4], "phrase": "Logos and posters"},
    {"class": [4], "phrase": "I need to re-brand with new designs and logos"},
    {"class": [1, 2], "phrase": "I want users to sign in to a website"},
    {"class": [1, 2], "phrase": "I need a database and webpage that takes emails"}
]

# Stems
stems = ['about', 'ap', 'apply', 'brand', 'databas', 'design', 'email', 'in',
         'inform', 'link', 'logo', 'new', 'pict', 'post', 'profil', 're-brand',
         'request', 'sav', 'send', 'sign', 'tak', 'that', 'us', 'web', 'webp',
         'websit']


if __name__ == "__main__":
    # Import dependencies
    from tagify import train, getDataSet

    # Make dataset out of data
    trainX, trainY = getDataSet(trainData, classes, stems)

    # Train and save model
    train(trainX, trainY, stems, "dummy", 1000, .02)