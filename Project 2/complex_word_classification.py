"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Neil Daterao

I have adhered to the Union College Honor Code in completing this project

Complete this file for parts 2-4 of the project.

"""

from collections import defaultdict
import gzip
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from syllables import count_syllables
from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate

def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    words, labels = load_file(data_file)
    y_pred = [1] * len(labels)
    evaluate(y_pred, labels)
    
    
### 2.2: Word length thresholding

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    
    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)
    
    max_word_length = max(len(word) for word in t_words)
    
    best_fscore = 0
    best_threshold = None
    
    for threshold_len in range(1, max_word_length+1):
        t_pred = [1 if len(word) >= threshold_len else 0 for word in t_words]
        f_score = get_fscore(t_pred, t_labels)
        
        if f_score > best_fscore:
            best_fscore = f_score
            best_threshold = threshold_len
            
    print(f"Best threshold found: {best_threshold} characters")
    
    # Evaluate on training and dev data
    print("\nPerformance on Training Data:")
    t_preds = [1 if len(word) >= best_threshold else 0 for word in t_words]
    evaluate(t_preds, t_labels)

   
    print("\nPerformance on Development Data:")
    d_preds = [1 if len(word) >= best_threshold else 0 for word in d_words]
    evaluate(d_preds, d_labels)


### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """

    t_words, t_labels = load_file(training_file)
    d_words, d_labels = load_file(development_file)
    
    word_frequencies = [counts[word] for word in t_words]
    min_freq, max_freq = min(word_frequencies), max(word_frequencies)
    
    
    #Log space is most efficient because word frequencies follow a power-law distribution, meaning most words are rare while a few are extremely common, 
    # and logarithmic scaling ensures we sample thresholds more effectively across this wide range.
    thresholds = np.logspace(np.log10(max(min_freq, 1)), np.log10(max_freq), num=50)
    
    best_threshold = None
    best_fscore = 0 
    
    #Follow similar procedure for word_length_threshold
    for threshold in thresholds:
        t_pred = [1 if len(word) >= threshold else 0 for word in t_words]
        f_score = get_fscore(t_pred, t_labels)
        
        if f_score > best_fscore:
            best_fscore = f_score
            best_threshold = threshold
            
    print(f"Best frequency threshold found: {int(best_threshold)} occurrences")

    # Evaluate on training and dev data
    print("\nPerformance on Training Data:")
    t_preds = [1 if counts[word] < best_threshold else 0 for word in t_words]
    evaluate(t_preds, t_labels)

    # Evaluate on development data using the best threshold
    print("\nPerformance on Development Data:")
    d_preds = [1 if counts[word] < best_threshold else 0 for word in d_words]
    evaluate(d_preds, d_labels)
    


### 3.1: Naive Bayes


def extract_features(counts, words):
    """
    Extracts word length and frequency features for a given list of words.
    Returns a 2D NumPy array where the first column is word length and the second column is word frequency.
    """
    lens = np.array([len(word) for word in words])
    freqs = np.array([counts[word] if word in counts else 1 for word in words])
    return np.array([lens,freqs]).T
    

def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    t_words, train_y = load_file(training_file)
    d_words, dev_y = load_file(development_file)
    
    train_x = extract_features(counts, t_words)
    dev_x = extract_features(counts, d_words)
    
    train_mean = train_x.mean(axis=0)
    train_sd = train_x.std(axis=0)
    
    
    #Normalize -> X_scaled = (X_original - mean) / std
    train_x = (train_x - train_mean) / train_sd  
    dev_x = (dev_x - train_mean) / train_sd 
    
    clf = GaussianNB()
    clf.fit(train_x, train_y)

    # Predict on training and dev data
    y_pred_train = clf.predict(train_x)
    print("\nPerformance on Training Data:")
    evaluate(y_pred_train, train_y)

    y_pred_dev = clf.predict(dev_x)
    print("\nPerformance on Development Data:")
    evaluate(y_pred_dev, dev_y)


### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    t_words, train_y = load_file(training_file)
    d_words, dev_y = load_file(development_file)
    
    train_x = extract_features(counts, t_words)
    dev_x = extract_features(counts, d_words)
    
    train_mean = train_x.mean(axis=0)
    train_sd = train_x.std(axis=0)
    
    
    #Normalize -> X_scaled = (X_original - mean) / std
    train_x = (train_x - train_mean) / train_sd  
    dev_x = (dev_x - train_mean) / train_sd 
    
    clf = LogisticRegression()
    clf.fit(train_x, train_y)

    # Predict on training and dev data
    y_pred_train = clf.predict(train_x)
    print("\nPerformance on Training Data:")
    evaluate(y_pred_train, train_y)

    y_pred_dev = clf.predict(dev_x)
    print("\nPerformance on Development Data:")
    evaluate(y_pred_dev, dev_y)
    
   

### 3.3: Build your own classifier

def error_analysis(y_pred, y_true, words):
    """Perform error analysis by displaying correctly and incorrectly classified words."""
    
    correct_predictions = []
    incorrect_predictions = []

    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:  # Correct case
            correct_predictions.append(words[i])
        else:  # Incorrect case
            incorrect_predictions.append((words[i], y_pred[i], y_true[i]))

    # Print analysis results
    print(f"\n **Correctly Classified Words (10 Examples):**")
    print(correct_predictions[:10])  

    print(f"\n **Misclassified Words (10 Examples):**")
    for word, pred, actual in incorrect_predictions[:10]: 
        print(f"Word: {word} | Predicted: {pred} | Actual: {actual}")


def extract_additional_features(words, counts, data_file):
    """
    Extracts word-based and sentence-based features. Word-based features include word length, frequency, syllable count, and synonym count. 
    Sentence-based features include sentence length, average word length, and average word frequency.
    """
    
    #Word based features
    lengths = np.array([len(word) for word in words])  
    freqs = np.array([counts[word] if word in counts else 1 for word in words]) 
    syllables = np.array([count_syllables(word) for word in words])  
    synonyms = np.array([len(wn.synsets(word)) for word in words])  

    
    sentences = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                sentence = line_split[3]  # Extract sentence from file
                sentences.append(sentence)
            i += 1

    # Sentence-based features
    sentence_lengths = np.array([len(sentence.split()) for sentence in sentences])  # Total words in sentence
    avg_word_length = np.array([sum(len(word) for word in sentence.split()) / len(sentence.split()) for sentence in sentences])
    avg_word_freq = np.array([
        sum(counts[word] if word in counts else 1 for word in sentence.split()) / len(sentence.split()) for sentence in sentences
    ])

    return np.array((lengths, freqs, syllables, synonyms, sentence_lengths, avg_word_length, avg_word_freq)).T
    



def my_classifier(training_file, development_file, counts):
    """
    Trains a Random Forest classifier using extracted word-based and sentence-based features. 
    The training and development data are normalized before fitting the model. 
    Evaluates performance on both datasets and conducts error analysis before returning the F-score.
    """
    t_words, train_y = load_file(training_file)
    d_words, dev_y = load_file(development_file)
    
    train_x = extract_additional_features(t_words, counts, training_file)
    dev_x = extract_additional_features(d_words, counts, development_file)
    
    train_mean = train_x.mean(axis=0)
    train_sd = train_x.std(axis=0)
    
    
    #Normalize -> X_scaled = (X_original - mean) / std
    train_x = (train_x - train_mean) / train_sd  
    dev_x = (dev_x - train_mean) / train_sd 
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    clf.fit(train_x, train_y)

    # Predict on training and dev data
    y_pred_train = clf.predict(train_x)
    print("\nPerformance on Training Data:")
    evaluate(y_pred_train, train_y)

    y_pred_dev = clf.predict(dev_x)
    print("\nPerformance on Development Data:")
    evaluate(y_pred_dev, dev_y)
    error_analysis(y_pred_dev, dev_y, d_words)
    f_score = get_fscore(y_pred_dev, dev_y)
    
    return f_score
    
    

def decision_tree_classifier(training_file, development_file, counts):
    """
    Trains a Decision Tree classifier using extracted features from the training and development files. 
    Data is normalized before training, and performance is evaluated on both datasets. 
    Conducts error analysis and returns the F-score of the development set predictions.
    """
    
    t_words, train_y = load_file(training_file)
    d_words, dev_y = load_file(development_file)
    
    train_x = extract_additional_features(t_words, counts, training_file)
    dev_x = extract_additional_features(d_words, counts, development_file)
    
    train_mean = train_x.mean(axis=0)
    train_sd = train_x.std(axis=0)
    
    
    #Normalize -> X_scaled = (X_original - mean) / std
    train_x = (train_x - train_mean) / train_sd  
    dev_x = (dev_x - train_mean) / train_sd 
    
    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(train_x, train_y)

    # Predict on training and dev data
    y_pred_train = clf.predict(train_x)
    print("\nPerformance on Training Data:")
    evaluate(y_pred_train, train_y)

    y_pred_dev = clf.predict(dev_x)
    print("\nPerformance on Development Data:")
    evaluate(y_pred_dev, dev_y)
    error_analysis(y_pred_dev, dev_y, d_words)
    f_score = get_fscore(y_pred_dev, dev_y)
    
    return f_score



def combine_files(training_file, development_file, output_file="combined_train_dev.txt"):
    """Physically combine training and development files into a single file, ensuring no duplicate headers."""
    with open(output_file, "w") as out_file:
       
        with open(training_file, "r") as train_f:
            header = train_f.readline()  
            out_file.write(header)
            out_file.write(train_f.read())  

        
        with open(development_file, "r") as dev_f:
            dev_f.readline()  # Skip header
            out_file.write(dev_f.read())  

    print(f"\n Combined file saved as `{output_file}`")
    return output_file 


def train_and_select_best_model(training_file, development_file, combined_file, test_file, counts, output_file="test_labels.txt"):
    """Train Decision Tree and Random Forest using existing functions, compare F-scores, and use the best model to predict test labels."""

    print("\n Evaluating Random Forest Classifier:")
    rf_fscore = my_classifier(training_file, development_file, counts)

    print("\n Evaluating Decision Tree Classifier:")
    dt_fscore = decision_tree_classifier(training_file, development_file, counts)

    print(f"\n Random Forest F-score: {rf_fscore:.2f}")
    print(f" Decision Tree F-score: {dt_fscore:.2f}")

    # Select the best model based on F-score
    if rf_fscore > dt_fscore:
        best_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
        print("\n Using Random Forest for final predictions.")
    else:
        best_model = DecisionTreeClassifier(max_depth=10, random_state=42)
        print("\n Using Decision Tree for final predictions.")

    combined_words, combined_labels = load_file(combined_file)
    
    # Load test words (unlabeled)
    test_words, _ = load_file(test_file)

    
    combined_x = extract_additional_features(combined_words, counts, combined_file)
    test_x = extract_additional_features(test_words, counts, test_file)

    #Normalize using the combined dataset statistics
    train_mean = combined_x.mean(axis=0)
    train_std = combined_x.std(axis=0)

    combined_x = (combined_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std  

    # Train the selected best model
    best_model.fit(combined_x, combined_labels)

    # Predict labels for the test set
    test_preds = best_model.predict(test_x)

    # Save predictions to file
    with open(output_file, "w") as f:
        for label in test_preds:
            f.write(str(label) + "\n")

    print(f"\n Predictions saved to {output_file}")




def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier (Random Forest)")
    print("-----------")
    my_classifier(training_file, development_file, counts)
    
    print("\nMy classifier (Decision Tree)")
    print("-----------")
    decision_tree_classifier(training_file, development_file, counts)

if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"
    combined_file = 'combined_train_dev.txt'

    print("Loading ngram counts ...")
    ngram_counts_file = "data/ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)

    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)

    ## YOUR CODE HERE
    # Train your best classifier, predict labels for the test dataset and write
    # the predicted labels to the text file 'test_labels.txt', with ONE LABEL
    # PER LINE
    combine_files(training_file, development_file)
    train_and_select_best_model(training_file, development_file, combined_file, test_file, counts, 'test_labels.txt')