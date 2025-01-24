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
    n_grams_list = []
    
    if c > 0:
        padded_text = start_pad(c) + text
    else:
        padded_text = text
    
    for i in range(c, len(padded_text)):
        if c > 0:
            context = padded_text[i - c:i]
        else:
            context = ''
        char = padded_text[i]
        n_grams_list.append((context, char))
    
    return n_grams_list
    

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
        self._context_length = c 
        self._k = k 
        self._vocab = set()
        self._ngram_counts = {}
        

    def get_vocab(self):
        """ Return the set of characters in the vocab """
        return self._vocab

    def update(self, text):
        """Update the model n-grams based on text.
        
        Parameters:
        - text (str): The text to update the model with.
        """
        # Update the vocabulary with characters from the text
        self._vocab.update(text)
        
        # Generate n-grams from the text
        ngram_list = ngrams(self._context_length, text)
        
        # Update n-gram counts
        for context, char in ngram_list:
            if context not in self._ngram_counts:
                self._ngram_counts[context] = {}
            if char not in self._ngram_counts[context]:
                self._ngram_counts[context][char] = 1
            else:
                self._ngram_counts[context][char] += 1

    def prob(self, context, char):
        """Return the probability of char appearing after context.
        
        Parameters:
        - context (str): A string of length c representing the context.
        - char (str): The character for which probability is calculated.
        
        Returns:
        - float: The probability of 'char' given 'context'.
        """
        V = len(self._vocab)
        if V == 0:
            return 0.0  
        
        if context not in self._ngram_counts:
            # Return uniform probability
            return 1.0 / V
        
        context_counts = self._ngram_counts[context]
        total = sum(context_counts.values())
        
        if char not in context_counts:
            # Character not seen with this context
            return 0.0
        else:
            return context_counts[char] / total

    def random_char(self, context):
        """Return a random character based on the given context and the 
        n-grams learned by this model.
        
        Parameters:
        - context (str): A string of length c representing the context.
        
        Returns:
        - str: A randomly selected character based on the probability distribution.
        """
        V = sorted(self._vocab)  # Lexicographically sorted vocabulary
        r = random.random()
        cumulative = 0.0

        for char in V:
            p = self.prob(context, char)
            cumulative += p
            if r < cumulative:
                return char
 
        return V[-1]

    def random_text(self, length):
        """Return a string of characters chosen at random using the random_char method.
        
        Parameters:
        - length (int): The number of random characters to generate.
        
        Returns:
        - str: A string of randomly generated characters.
        """
        if self._context_length > 0:
            context = start_pad(self._context_length)
        else:
            context = ''
        
        result = []
        
        for _ in range(length):
            char = self.random_char(context)
            if char is None:
                break  # If vocab is empty, stop generating
            result.append(char)
            if self._context_length > 0:
                context = context[1:] + char  # Update context by appending new char
            # If c=0, context remains ''
        
        return ''.join(result)
    
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

# -------------------------------
# Test Functions
# -------------------------------
def test_ngrams_c1():
    print("Testing ngrams Function - Context Length 1:")
    test_text = 'abc'
    expected = [('~', 'a'), ('a', 'b'), ('b', 'c')]
    actual = ngrams(1, test_text)
    print(f"ngrams(1, '{test_text}') = {actual}")
    print("Test Passed:", actual == expected)
    print()

def test_ngrams_c2():
    print("Testing ngrams Function - Context Length 2:")
    test_text = 'abc'
    expected = [('~~', 'a'), ('~a', 'b'), ('ab', 'c')]
    actual = ngrams(2, test_text)
    print(f"ngrams(2, '{test_text}') = {actual}")
    print("Test Passed:", actual == expected)
    print()

def test_vocab_after_abab():
    print("Testing NgramModel Class - Vocabulary after 'abab':")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    expected = {'a', 'b'}
    actual = m.get_vocab()
    print("Vocabulary after 'abab':", actual)
    print("Test Passed:", actual == expected)
    print()

def test_vocab_after_abcd():
    print("Testing NgramModel Class - Vocabulary after 'abcd':")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    expected = {'a', 'b', 'c', 'd'}
    actual = m.get_vocab()
    print("Vocabulary after 'abcd':", actual)
    print("Test Passed:", actual == expected)
    print()

def test_prob_a_b():
    print("Testing NgramModel Class - Probability P('b' | 'a'):")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    expected = 1.0
    actual = m.prob('a', 'b')
    print("Probability P('b' | 'a'):", actual)
    print("Test Passed:", actual == expected)
    print()

def test_prob_tilde_c():
    print("Testing NgramModel Class - Probability P('c' | '~'):")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    expected = 0.0
    actual = m.prob('~', 'c')
    print("Probability P('c' | '~'):", actual)
    print("Test Passed:", actual == expected)
    print()

def test_prob_b_c():
    print("Testing NgramModel Class - Probability P('c' | 'b'):")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    expected = 0.5
    actual = m.prob('b', 'c')
    print("Probability P('c' | 'b'):", actual)
    print("Test Passed:", actual == expected)
    print()

def test_random_char():
    print("Testing random_char Method:")
    m = NgramModel(c=0, k=0)
    m.update('abab')
    m.update('abcd')
    
    # Set seed for reproducibility
    random.seed(1)
    
    # Generate 25 random characters based on the model
    random_chars = [m.random_char('') for _ in range(4)]
    print("Random characters:", random_chars)
    
    # Expected Output:
    expected = ['a', 'c', 'c', 'a', 'b', 'b', 'b', 'c', 'a', 'a', 'c', 'b', 'c', 'a', 'b', 'b', 'a',
                'd', 'd', 'a', 'a', 'b', 'd', 'b', 'a']
    
    print("Expected characters:", expected)
    print("Test Passed:", random_chars == expected)
    print()
    

def test_random_text():
    print("Testing random_text Method:")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    
    # Set seed for reproducibility
    random.seed(1)
    
    # Generate 25 random characters based on the model
    output = m.random_text(25)
    print("Random text:", output)
    
    # Expected Output:
    expected = 'abcdbabcdabababcdddabcdba'
    
    print("Expected text:", expected)
    print("Test Passed:", output == expected)
    print()
    

def test_additional_vocab_c1():
    print("Running Additional Test Cases - Vocabulary for c=1:")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    expected = {'a', 'b', 'c', 'd'}
    actual = m.get_vocab()
    print("Vocabulary for c=1:", actual)
    print("Test Passed:", actual == expected)
    print()

def test_additional_prob_a_tilde():
    print("Running Additional Test Cases - Probability P('a' | '~'):")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    expected = 1.0  # Since only 'a' follows '~'
    actual = m.prob('~', 'a')
    print("Probability P('a' | '~'):", actual)
    print("Test Passed:", actual == expected)
    print()

def test_additional_prob_d_c():
    print("Running Additional Test Cases - Probability P('d' | 'c'):")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    expected = 1.0  # 'c' is only followed by 'd'
    actual = m.prob('c', 'd')
    print("Probability P('d' | 'c'):", actual)
    print("Test Passed:", actual == expected)
    print()

def test_additional_prob_e_c():
    print("Running Additional Test Cases - Probability P('e' | 'c'):")
    m = NgramModel(c=1, k=0)
    m.update('abab')
    m.update('abcd')
    expected = 0.0  # 'e' not in vocab
    actual = m.prob('c', 'e')
    print("Probability P('e' | 'c'):", actual)
    print("Test Passed:", actual == expected)
    print()
    
def test_shakespeare():
    m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 2)
    print('---------------------')
    print("Test 1:", m.random_text(250))
    print('---------------------')
    m1 = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
    print('---------------------')
    print("Test 2:", m1.random_text(250))
    print('---------------------')
    m2 = create_ngram_model(NgramModel, 'shakespeare_input.txt', 4)
    print('---------------------')
    print("Test 3:", m2.random_text(250))
    print('---------------------')
    m3 = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
    print('---------------------')
    print("Test 4:", m3.random_text(250))
    print('---------------------')


# -------------------------------
# Function to Run All Tests
# -------------------------------
def run_all_tests():
    print("======================================")
    print("Starting All Test Cases")
    print("======================================\n")
    
    # Testing ngrams function
    test_ngrams_c1()
    test_ngrams_c2()
    
    # Testing NgramModel class vocab and probabilities
    test_vocab_after_abab()
    test_vocab_after_abcd()
    test_prob_a_b()
    test_prob_tilde_c()
    test_prob_b_c()
    
    # Testing random_char method
    test_random_char()
    
    #Testing Random Text
    test_random_text()
    
    # Running additional test cases
    test_additional_vocab_c1()
    test_additional_prob_a_tilde()
    test_additional_prob_d_c()
    test_additional_prob_e_c()
    
    
    test_shakespeare()
    print("======================================")
    print("All Test Cases Completed")
    print("======================================")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    run_all_tests()


