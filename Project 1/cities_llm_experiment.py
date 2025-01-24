import os
import math
from collections import defaultdict
from ngram_lm import NgramModel, create_ngram_model, start_pad, COUNTRY_CODES
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

'''
Experiment: Predicting a Country that a city is in

Author: Neil Daterao
CSC-306
'''

################################################################################
# Prediction and Evaluation Functions
################################################################################

def predict_country(city, models, c):
    """
    Predict the country of a given city by computing log probabilities under each model.

    Parameters:
    - city (str): The name of the city (preprocessed).
    - models (dict): A dictionary mapping country codes to trained n-gram models.
    - c (int): The context length (n-1).

    Returns:
    - str: The predicted country code.
    """
    log_probs = {}
    for country, model in models.items():
        log_prob = 0.0
        context = start_pad(c)
        for char in city:
            prob = model.prob(context, char)
            if prob == 0.0:
                log_prob = float('-inf')
                break
            log_prob += math.log(prob)
            if c > 0:
                context = context[1:] + char
        log_probs[country] = log_prob
    
    # Select the country with the highest log probability
    predicted_country = max(log_probs, key=log_probs.get)
    return predicted_country

def evaluate_models(models, val_data, c):
    """
    Evaluate all models on the validation dataset and compute accuracy.

    Parameters:
    - models (dict): A dictionary mapping country codes to trained n-gram models.
    - val_data (dict): A dictionary mapping country codes to lists of city names.
    - c (int): The context length (n-1).

    Returns:
    - float: Overall accuracy as a percentage.
    - dict: Per-country accuracy percentages.
    """
    correct = 0
    total = 0
    per_country_correct = defaultdict(int)
    per_country_total = defaultdict(int)

    for correct_country, cities in val_data.items():
        for city in cities:
            predicted_country = predict_country(city, models, c)
            if predicted_country == correct_country:
                correct += 1
                per_country_correct[correct_country] += 1
            per_country_total[correct_country] += 1
            total += 1

    overall_accuracy = (correct / total) * 100 if total > 0 else 0.0

    per_country_accuracy = {}
    for country in per_country_total:
        acc = (per_country_correct[country] / per_country_total[country]) * 100 if per_country_total[country] > 0 else 0.0
        per_country_accuracy[country] = acc

    return overall_accuracy, per_country_accuracy

################################################################################
# Experimentation Functions
################################################################################

def train_and_evaluate(train_dir, val_dir, country_codes, c_values, k_values):
    """
    Perform grid search over different c and k values, train models, evaluate them,
    and identify the best configuration.

    Parameters:
    - train_dir (str): Path to the training data directory.
    - val_dir (str): Path to the validation data directory.
    - country_codes (list): List of country codes.
    - c_values (list): List of context lengths to experiment with.
    - k_values (list): List of smoothing parameters to experiment with.

    Returns:
    - dict: A dictionary containing accuracy results for each (c, k) combination.
    """
    # Load Validation Data
    print("\nLoading validation data...")
    val_data = {}
    for country in country_codes:
        val_file = os.path.join(val_dir, f"{country}.txt")
        if not os.path.isfile(val_file):
            print(f"Validation file for country '{country}' not found at '{val_file}'. Skipping.")
            continue
        with open(val_file, 'r', encoding='utf-8', errors='ignore') as f:
            cities = [line.strip().lower() for line in f if line.strip()]
            val_data[country] = cities
        print(f"Loaded {len(cities)} validation cities for country '{country}'")

    results = {}

    # Iterate over all combinations of c and k, try to get the best possible model
    for c in c_values:
        for k in k_values:
            print(f"\n--- Training with c={c} (n={c+1}), k={k} ---")
            models = {}
            for country in country_codes:
                train_file = os.path.join(train_dir, f"{country}.txt")
                if not os.path.isfile(train_file):
                    print(f"Training file for country '{country}' not found at '{train_file}'. Skipping.")
                    continue
               
                # Create and train the n-gram model
                model = create_ngram_model(NgramModel, train_file, c, k)
                models[country] = model
                print(f"Trained model for country '{country}' with vocabulary size {len(model.get_vocab())}")

            # Evaluate the current set of models
            print(f"\nEvaluating models with c={c} and k={k}...")
            overall_accuracy, per_country_accuracy = evaluate_models(models, val_data, c)
            print(f"Overall Validation Accuracy: {overall_accuracy:.2f}%")
            print("Per-Country Validation Accuracy:")
            for country in country_codes:
                if country in per_country_accuracy:
                    print(f"  {country}: {per_country_accuracy[country]:.2f}%")
                else:
                    print(f"  {country}: No validation data available.")

            # Store the results
            results[(c, k)] = {
                'overall_accuracy': overall_accuracy,
                'per_country_accuracy': per_country_accuracy
            }

    return results

################################################################################
# Optional: Visualization Function
################################################################################

def plot_results(results, c_values, k_values):
    """
    Plots a heatmap of overall accuracies for different c and k values.

    Parameters:
    - results (dict): Dictionary containing accuracy results for each (c, k) combination.
    - c_values (list): List of context lengths.
    - k_values (list): List of smoothing parameters.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Prepare data for the heatmap
    data = []
    for c in c_values:
        row = []
        for k in k_values:
            acc = results.get((c, k), {}).get('overall_accuracy', 0)
            row.append(acc)
        data.append(row)

    df = pd.DataFrame(data, index=[f'c={c} (n={c+1})' for c in c_values],
                      columns=[f'k={k}' for k in k_values])

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Validation Accuracy for Different c and k Values")
    plt.ylabel("Context Length (c)")
    plt.xlabel("Smoothing Parameter (k)")
    plt.tight_layout()
    plt.show()

def main():
  
    train_dir = 'train'
    val_dir = 'val'
    c_values = [1, 2, 3]  # Corresponding to bigrams, trigrams, and four-grams
    k_values = [0, 0.5, 1, 2]  # Smoothing parameters

    print("Starting grid search over different c and k values...")
    results = train_and_evaluate(train_dir, val_dir, COUNTRY_CODES, c_values, k_values)

    #Get best config
    best_accuracy = -1
    best_config = None

    for config, metrics in results.items():
        if metrics['overall_accuracy'] > best_accuracy:
            best_accuracy = metrics['overall_accuracy']
            best_config = config

    if best_config:
        best_c, best_k = best_config
        print(f"\nBest Configuration: c={best_c} (n={best_c+1}), k={best_k} with Overall Accuracy={best_accuracy:.2f}%")
    else:
        print("\nNo valid configurations found.")

    try:
        plot_results(results, c_values, k_values)
    except ImportError:
        print("\nMatplotlib or Seaborn not installed. Skipping the plotting of results.")

if __name__ == "__main__":
    main()
