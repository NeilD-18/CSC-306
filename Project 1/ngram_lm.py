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
            return 0.0  # Avoid division by zero 

        if self._k > 0:
            # Add-k smoothing is applied
            if context not in self._ngram_counts:
                # Novel context: assign equal probability to all characters
                numerator = self._k
                denominator = self._k * V
                return numerator / denominator
            
            context_counts = self._ngram_counts[context]
            total = sum(context_counts.values()) + self._k * V  # Adjusted total with smoothing
            
            if char not in context_counts:
                count = 0
            else:
                count = context_counts[char]
            
            numerator = count + self._k
            return numerator / total
        else:
            
            if context not in self._ngram_counts:
                # Uniform Probability
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
 
        return None #Should maybe return something else here but fine for now

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
                break  # Break if vocab is missing
            result.append(char)
            if self._context_length > 0:
                context = context[1:] + char  # Update context by appending new char
           
        
        return ''.join(result)
    
    def perplexity(self, text):
        """Return the perplexity of text based on the n-grams learned by this
        model.
        
        Perplexity is defined as:
        Perplexity(text) = exp(- (1/N) * sum(log(P(w_i | context_i))))
        
        If any P(w_i | context_i) == 0, return float('inf').
        
        Parameters:
        - text (str): The text for which perplexity is calculated.
        
        Returns:
        - float: The perplexity value.
        """
        N = 0
        log_prob_sum = 0.0
        
        if self._context_length > 0:
            context = start_pad(self._context_length)
        else:
            context = ''
        
        for char in text:
            prob = self.prob(context, char)
            if prob == 0.0:
                return float('inf')
            log_prob_sum += math.log(prob)
            N += 1
            if self._context_length > 0:
                context = context[1:] + char
        
        if N == 0:
            return float('inf')  # Undefined perplexity for empty text
        
        avg_log_prob = log_prob_sum / N
        perplexity = math.exp(-avg_log_prob)
        return perplexity
################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    """An n-gram model with interpolation."""
    
    def __init__(self, c, k):
        """
        Initialize the NgramModelWithInterpolation.
        
        Parameters:
        - c (int): The highest order n-gram
        - k (float): The smoothing parameter for add-k smoothing.
        """
        super().__init__(c, k)
        
        # Create separate NgramModel instances for each order from 0 (unigram) to c
        self.ngram_models = [NgramModel(order, k) for order in range(c + 1)]  
        
        # Initialize lambdas to equal weights
        self.lambdas = [1.0 / (c + 1) for _ in range(c + 1)]
    
    def set_lambdas(self, lambdas):
        """
        Set the lambda weights for interpolation.
        
        Parameters:
        - lambdas (list of float): A list of lambda weights. Length should be c + 1.
                                   The weights should sum to 1.
    
        """
        if len(lambdas) != self._context_length + 1:
            raise ValueError(f"Number of lambdas must be {self._context_length + 1}.")
        if not math.isclose(sum(lambdas), 1.0, rel_tol=1e-9):
            raise ValueError("Sum of lambdas must be 1.")
        self.lambdas = lambdas
    
    def get_vocab(self):
        """
        Return the combined vocabulary from all n-gram models.
        
        Returns:
        - set: The set of unique characters in the combined vocabulary.
        """
        combined_vocab = set()
        for model in self.ngram_models:
            combined_vocab.update(model.get_vocab())
        return combined_vocab
    
    def update(self, text):
        """
        Update all constituent n-gram models with the provided text.
        
        Parameters:
        - text (str): The text to update the models with.
        """
        for model in self.ngram_models:
            model.update(text)
        
        # Update the combined vocabulary in the base class
        self._vocab.update(self.get_vocab())
    
    def prob(self, context, char):
        """
        Compute the interpolated probability of a character given its context.
        
        Parameters:
        - context (str): The context string of length up to c.
        - char (str): The character for which probability is calculated.
        
        Returns:
        - float: The interpolated probability of 'char' given 'context'.
        """
        interpolated_prob = 0.0
        
        # Iterate through each n-gram model and accumulate the weighted probabilities
        for order, model in enumerate(self.ngram_models):
            if order == 0:
                sub_context = ''
            else:
                if len(context) >= order:
                    sub_context = context[-order:]
                else:
                    sub_context = start_pad(order - len(context)) + context
                    
            p = model.prob(sub_context, char)
            interpolated_prob += self.lambdas[order] * p
        
        return interpolated_prob

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
    random_chars = [m.random_char('') for _ in range(25)]
    print("Random characters:", random_chars)
    
    # Expected Output:
    expected = ['a', 'c', 'c', 'a', 'b', 'b', 'b', 'c', 'a', 'a', 'c', 'b', 'c', 'a', 'b', 'b', 'a',
                'd', 'd', 'a', 'a', 'b', 'd', 'b', 'a']
    
    print("Expected characters:", expected)
    print("Test Passed:", random_chars == expected)
    print()
    

def test_random_text():
    """Test the random_text method with a specific seed."""
    print("Testing random_text Method:")
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    
    # Set seed for reproducibility
    random.seed(1)
    
    # Generate 25 random characters based on the model
    generated = m.random_text(25)
    expected = 'abcdbabcdabababcdddabcdba'
    print(f"Generated Text: {generated}")
    print(f"Expected Text:  {expected}")
    print("Test Passed:", generated == expected)
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
    

def test_perplexity():
    """Test the perplexity method."""
    print("Testing perplexity Method:")
    m = NgramModel(1, 0)
    m.update('abab')
    m.update('abcd')
    
 
    test_text1 = 'abcd'
    expected_pp1 = 1.189207115002721
    actual_pp1 = m.perplexity(test_text1)
    print(f"Perplexity('{test_text1}') = {actual_pp1}")
    print("Test Passed:", math.isclose(actual_pp1, expected_pp1, rel_tol=1e-9))
    print()
    
  
    test_text2 = 'abca'
    expected_pp2 = float('inf')
    actual_pp2 = m.perplexity(test_text2)
    print(f"Perplexity('{test_text2}') = {actual_pp2}")
    print("Test Passed:", actual_pp2 == expected_pp2)
    print()
    
    
    test_text3 = 'abcda'
    expected_pp3 = 1.515716566510398
    actual_pp3 = m.perplexity(test_text3)
    print(f"Perplexity('{test_text3}') = {actual_pp3}")
    print("Test Passed:", math.isclose(actual_pp3, expected_pp3, rel_tol=1e-9))
    print()


def test_add_k_smoothing():
    """Test the add-k smoothing implementation in the prob method."""
    print("Testing Add-k Smoothing in prob Method:")
    m = NgramModel(1, 1)
    m.update('abab')
    m.update('abcd')
    
  
    context1 = 'a'
    char1 = 'a'
    expected_prob1 = 0.14285714285714285
    actual_prob1 = m.prob(context1, char1)
    print(f"P('{char1}' | '{context1}') = {actual_prob1}")
    print("Test Passed:", math.isclose(actual_prob1, expected_prob1, rel_tol=1e-9))
    print()
    
 
    context2 = 'a'
    char2 = 'b'
    expected_prob2 = 0.5714285714285714
    actual_prob2 = m.prob(context2, char2)
    print(f"P('{char2}' | '{context2}') = {actual_prob2}")
    print("Test Passed:", math.isclose(actual_prob2, expected_prob2, rel_tol=1e-9))
    print()
    
    
    context3 = 'c'
    char3 = 'd'
    expected_prob3 = 0.4
    actual_prob3 = m.prob(context3, char3)
    print(f"P('{char3}' | '{context3}') = {actual_prob3}")
    print("Test Passed:", math.isclose(actual_prob3, expected_prob3, rel_tol=1e-9))
    print()
    
    context4 = 'd'
    char4 = 'a'
    expected_prob4 = 0.25
    actual_prob4 = m.prob(context4, char4)
    print(f"P('{char4}' | '{context4}') = {actual_prob4}")
    print("Test Passed:", math.isclose(actual_prob4, expected_prob4, rel_tol=1e-9))
    print()


def test_interpolation_order1():
    """Test the NgramModelWithInterpolation with order=1 (bigrams)."""
    print("Testing NgramModelWithInterpolation with Order=1:")
    
    # Initialize the model with context length 1 (bigrams) and k=0 (no smoothing)
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    
 
    context1 = 'a'
    char1 = 'a'
    expected_prob1 = 0.25
    actual_prob1 = m.prob(context1, char1)
    print(f"P('{char1}' | '{context1}') = {actual_prob1}")
    print("Test Passed:", math.isclose(actual_prob1, expected_prob1, rel_tol=1e-9))
    print()
    

    context2 = 'a'
    char2 = 'b'
    expected_prob2 = 0.75
    actual_prob2 = m.prob(context2, char2)
    print(f"P('{char2}' | '{context2}') = {actual_prob2}")
    print("Test Passed:", math.isclose(actual_prob2, expected_prob2, rel_tol=1e-9))
    print()

def test_interpolation_order2():
    """Test the NgramModelWithInterpolation with order=2 (trigrams)."""
    print("Testing NgramModelWithInterpolation with Order=2:")
    
    # Initialize the model with context length 2 (trigrams) and k=1 (add-1 smoothing)
    m = NgramModelWithInterpolation(2, 1)
    m.update('abab')
    m.update('abcd')
    
    context1 = '~a'
    char1 = 'b'
    expected_prob1 = 0.4682539682539682
    actual_prob1 = m.prob(context1, char1)
    print(f"P('{char1}' | '{context1}') = {actual_prob1}")
    print("Test Passed:", math.isclose(actual_prob1, expected_prob1, rel_tol=1e-9))
    print()
    

    context2 = 'ba'
    char2 = 'b'
    expected_prob2 = 0.4349206349206349
    actual_prob2 = m.prob(context2, char2)
    print(f"P('{char2}' | '{context2}') = {actual_prob2}")
    print("Test Passed:", math.isclose(actual_prob2, expected_prob2, rel_tol=1e-9))
    print()

    context3 = '~c'
    char3 = 'd'
    expected_prob3 = 0.27222222222222225
    actual_prob3 = m.prob(context3, char3)
    print(f"P('{char3}' | '{context3}') = {actual_prob3}")
    print("Test Passed:", math.isclose(actual_prob3, expected_prob3, rel_tol=1e-9))
    print()
    

    context4 = 'bc'
    char4 = 'd'
    expected_prob4 = 0.3222222222222222
    actual_prob4 = m.prob(context4, char4)
    print(f"P('{char4}' | '{context4}') = {actual_prob4}")
    print("Test Passed:", math.isclose(actual_prob4, expected_prob4, rel_tol=1e-9))
    print()

def test_interpolation_custom_lambdas():
    """Test the NgramModelWithInterpolation with custom lambda values."""
    print("Testing NgramModelWithInterpolation with Custom Lambdas:")
    
    # Initialize the model with context length 1 (bigrams) and k=1 (add-1 smoothing)
    m = NgramModelWithInterpolation(1, 1)
    m.update('abab')
    m.update('abcd')
    
    # Set custom lambdas, e.g., λ1=0.7 (bigram), λ2=0.3 (unigram)
    m.set_lambdas([0.3, 0.7])
    
    # Test Case: P('a' | 'a') with custom lambdas
    context1 = 'a'
    char1 = 'a'
    expected_prob1 = 0.3 * m.ngram_models[0].prob('', 'a') + 0.7 * m.ngram_models[1].prob('a', 'a')
    actual_prob1 = m.prob(context1, char1)
    print(f"P('{char1}' | '{context1}') = {actual_prob1} (Expected: {expected_prob1})")
    print("Test Passed:", math.isclose(actual_prob1, expected_prob1, rel_tol=1e-9))
    print()

def test_random_text_interpolation():
    """Test the random_text method of NgramModelWithInterpolation."""
    print("Testing random_text Method with Interpolation:")
    
    # Initialize the model with context length 1 (bigrams) and k=0 (no smoothing)
    m = NgramModelWithInterpolation(1, 0)
    m.update('abab')
    

    m.set_lambdas([0.5, 0.5])
    
    random.seed(1)
    
    # Generate 25 random characters based on the model
    generated = m.random_text(25)
    expected = 'abcdbabcdabababcdddabcdba'
    print(f"Generated Text: {generated}")
    print(f"Expected Text:  {expected}")
    print("Test Passed:", generated == expected)
    print()


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
    
    
    #Testing Perplexity
    test_perplexity()
    

    #Testing K-smoothing
    test_add_k_smoothing()
    
    # Running additional test cases
    test_additional_vocab_c1()
    test_additional_prob_a_tilde()
    test_additional_prob_d_c()
    test_additional_prob_e_c()
    
    
    # Testing interpolation with order=1
    test_interpolation_order1()
    
    # Testing interpolation with order=2
    test_interpolation_order2()
    
    # Testing interpolation with custom lambdas
    test_interpolation_custom_lambdas()
    
    
    test_shakespeare()
    print("======================================")
    print("All Test Cases Completed")
    print("======================================")
    

# -------------------------------
# Experiment Functions
# -------------------------------
def experiment_shakespeare():
    print("\nExperiment: Generating Shakespeare Text\n")
    
    # Test with different n-gram orders
    for n in [2, 3, 4, 7]:
        print(f"\nUsing n-gram model with n={n}:\n")
        model = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=n, k=0)
        generated_text = model.random_text(250)
        print(generated_text)
        print("-" * 80)
        
def experiment_different_n_values():
    print("\nExperiment: Perplexity with Different n Values\n")
    
    for n in [1, 2, 3, 4, 5]:
        model = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=n, k=0)
        perplexity = model.perplexity("To be, or not to be, that is the question.")
        print(f"n={n}, Perplexity: {perplexity}")


def experiment_different_k_values():
    print("\nExperiment: Perplexity with Different k Values (Smoothing)\n")
    
    for k in [0, 0.1, 0.5, 1, 2]:
        model = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=3, k=k)
        perplexity = model.perplexity("To be, or not to be, that is the question.")
        print(f"k={k}, Perplexity: {perplexity}")

def experiment_different_lambda_values():
    print("\nExperiment: Interpolation with Different Lambda Values\n")
    
    model = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', c=2, k=0)
    
    # Test different lambda configurations
    lambda_configs = [
        [1.0, 0.0],  # Only unigram
        [0.0, 1.0],  # Only bigram
        [0.5, 0.5],  # Equal weights
        [0.7, 0.3],  # Skewed to bigram
    ]
    
    for lambdas in lambda_configs:
        model.set_lambdas(lambdas)
        perplexity = model.perplexity("To be, or not to be, that is the question.")
        print(f"Lambdas={lambdas}, Perplexity: {perplexity}")


def experiment_shakespeare():
    print("\nExperiment: Generating Shakespeare Text\n")
    
    # Test with different n-gram orders
    for n in [2, 3, 4, 7]:
        print(f"\nUsing n-gram model with n={n}:\n")
        model = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=n, k=0)
        generated_text = model.random_text(250)
        print(generated_text)
        print("-" * 80)

def experiment_different_n_values():
    print("\nExperiment: Perplexity with Different n Values\n")
    
    for n in [1, 2, 3, 4, 5]:
        model = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=n, k=1)
        perplexity = model.perplexity("To be, or not to be, that is the question.")
        print(f"n={n}, Perplexity: {perplexity}")


def experiment_different_k_values():
    print("\nExperiment: Perplexity with Different k Values (Smoothing)\n")
    
    for k in [0, 0.1, 0.5, 1, 2]:
        model = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=3, k=k)
        perplexity = model.perplexity("To be, or not to be, that is the question.")
        print(f"k={k}, Perplexity: {perplexity}")

def experiment_different_lambda_values():
    print("\nExperiment: Interpolation with Different Lambda Values\n")
    
    # Initialize the model with context length 2 (trigrams) and k=0 (no smoothing)
    model = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', c=2, k=0)
    
    # Updated lambda configurations with 3 values each
    lambda_configs = [
        [1.0, 0.0, 0.0],  # Only unigram
        [0.0, 1.0, 0.0],  # Only bigram
        [0.0, 0.0, 1.0],  # Only trigram
        [0.33, 0.33, 0.34],  # Approximately equal weights
        [0.1, 0.2, 0.7],  # Skewed towards trigram
    ]
    
    for lambdas in lambda_configs:
        try:
            model.set_lambdas(lambdas)
            perplexity = model.perplexity("To be, or not to be, that is the question.")
            print(f"Lambdas={lambdas}, Perplexity: {perplexity}")
        except ValueError as e:
            print(f"Failed to set lambdas {lambdas}: {e}")

def experiment_cross_domain_evaluation():
    print("\nExperiment: Cross-Domain Evaluation\n")
    
    # Train on Shakespeare plays and evaluate on sonnets and NYT articles
    training_model = create_ngram_model(NgramModel, 'shakespeare_input.txt', c=3, k=1)
    
    # Test on similar domain
    with open('shakespeare_sonnets.txt', 'r') as sonnets_file:
        sonnets_text = sonnets_file.read()
    sonnets_perplexity = training_model.perplexity(sonnets_text)
    print(f"Perplexity on Shakespeare Sonnets: {sonnets_perplexity}")
    
    # Test on different domain
    with open('nytimes_article.txt', 'r', encoding='utf-8', errors='ignore') as nyt_file:
        nyt_text = nyt_file.read()
    nyt_perplexity = training_model.perplexity(nyt_text)
    print(f"Perplexity on NY Times Articles: {nyt_perplexity}")
    
    
    
def run_all_experiments():
    experiment_shakespeare()
    experiment_different_n_values()
    experiment_different_k_values()
    experiment_different_lambda_values()
    experiment_cross_domain_evaluation()

if __name__ == "__main__":
    run_all_tests()
    run_all_experiments()



