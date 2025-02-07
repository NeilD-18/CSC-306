"""Evaluation Metrics

Author: Kristina Striegnitz and Neil Daterao

I have adhered to the Union College Honor Code in completing this project

Complete this file for part 1 of the project.
"""

def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    correct = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true:
            correct += 1
    accuracy = correct/(len(y_pred))
    return accuracy if len(y_pred) > 0 else 0.0

def get_precision(y_pred, y_true, label=1):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    true_pos = 0
    false_pos = 0
    
    for pred, true in zip(y_pred, y_true):
        if pred == label and pred == true:
            true_pos += 1
        elif pred == label and pred != true:
            false_pos += 1
        else:
            continue
    
    if (true_pos + false_pos) == 0:
        return 0.0
    
    
    
    precision = true_pos / (true_pos + false_pos)
    
    return precision


def get_recall(y_pred, y_true, label=1):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    true_pos = 0
    false_neg = 0 
    
    for pred, true in zip(y_pred, y_true):
        if pred == label and pred == true:
            true_pos += 1
        elif pred !=label and true == label:
            false_neg += 1
    
    if (true_pos + false_neg) == 0:
        return 0.0 
  
    recall = true_pos / (true_pos + false_neg)
    return recall 


def get_fscore(y_pred, y_true, label=1):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    precision = get_precision(y_pred, y_true, label)
    recall = get_recall(y_pred, y_true, label)
    
    if (precision + recall == 0):
        return 0.0
    
    fscore = (2*precision*recall)  / (precision + recall)
    
    return fscore


def evaluate(y_pred, y_true, label=1):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    
    print(f"Accuracy: {get_accuracy(y_pred,y_true) * 100:.0f}%")
    print(f"Precision: {get_precision(y_pred,y_true, label) * 100:.0f}%")
    print(f"Recall: {get_recall(y_pred,y_true, label) * 100:.0f}%")
    print(f"F-score: {get_fscore(y_pred,y_true, label) * 100:.0f}%")
    
    


