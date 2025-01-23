import math, random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    """Return a padding string of length c to append to the front of text
    as a pre-processing step to building n-grams. c = n-1.
    """
    return '~' * c

def ngrams(c, text):
    """Return the ngrams of the text as a list of tuples.

    c is the length of the context (e.g. 1 for bigrams). The first
    element of each tuple is is the context string and the second is
    the character.
    """
    pass

def create_ngram_model(model_class, path, c=2, k=0):
    """Create and returns a new n-gram model trained on the entire text
    found in the path file.
    """
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, c=2, k=0):
    """Create and returns a new n-gram model trained line by line on the
    text found in the path file. 
    """
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    """ A basic n-gram model using add-k smoothing """

    def __init__(self, c, k):
        pass

    def get_vocab(self):
        """ Return the set of characters in the vocab """
        pass

    def update(self, text):
        """ Update the model n-grams based on text """
        pass

    def prob(self, context, char):
        """ Return the probability of char appearing after context """
        pass

    def random_char(self, context):
        """ Return a random character based on the given context and the 
        n-grams learned by this model 
        """
        pass

    def random_text(self, length):
        """Return a text of the specified character length based on the
        n-grams learned by this model
        """
        pass

    def perplexity(self, text):
        """Return the perplexity of text based on the n-grams learned by this
        model
        """
        pass

################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    """ An n-gram model with interpolation """

    def __init__(self, c, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text):
        pass

    def prob(self, context, char):
        pass

################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.
