# tagging-api
A Communicode NLP microservice that pulls skill keywords from description blocks. 

## Usage Instructions
#### System Requirements
As of yet, only Windows 10 has been tested.
Must have python 3.5.x with Flask, numpy, tensorflow, and NLTK
+ Download source in a working directory
+ Run API from terminal with command `python app.py`
+ Send a JSON POST request to `http://127.0.0.1:5000/tagify/v1.0/tagify/<model name>`
The JSON request must have a `description` field with a string representing the desired text to be classified.
Models available are `dummy`, `dummyDeep`, and `dummyH20` (replace name in `<model name>`).
I recomend https://insomnia.rest/ as a REST API client.

## The Algorithm
#### Preprocessing Training Data
A given phrase is first tokenized and stemmed using the Lancaster Stemmer from the "Natural Language Toolkit" by the NLTK Project. Certain characters and phrases are removed from the phrase like "and", "we", and "?". The total ordered list of these stemmed tokens is called the "stems".
#### Phrases to Vectors
Currently, I am using the Bag-Of-Words (BOW) technique, though, I will be testing Word2Vec. To implement BOW, each phrase is tokenized and stemmed. A vector is created with a zero if a stem in the phrase is not in the list of stems and a one if it is. The original list of stems is ordered and the vector of zeros and ones corresponds exactly with that order. If there are tokens in the phrase that weren't in the training data, and thus not in the list of stems, they are ignored.
### Classes to Vectors
Each class in the training data is given an index (ie. "HTML" -> 1). The label for any given piece of data is a one-hot (though there may be more than one class).
#### Neural Network
The model uses a one layer nerual network that predicts the class vector for a given phrase vector. The sigmoid function is used to tranform any output to a number between zero and one. These outputs between zero and one are the confidences for each class.

## The API
This system uses the Flask web library for the API. A POST request is sent to the API with a given non-profit description. The API returns a JSON file of class names and their given confidences.

## Results
Even on only 6 items of training data, the system performed extremely accurately and quickly.

## TODO
+ Test Word2Vec
+ Train on actual data
