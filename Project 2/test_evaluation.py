""" Testing File for Evaluation Metrics

Author: Neil Daterao

I have adhered to the Union College Honor Code in completing this project

"""

from evaluation import get_accuracy, get_fscore, get_precision, get_recall


def test_get_accuracy():
    print("Testing get_accuracy...\n")

    y_pred = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    expected = 0.7
    actual = get_accuracy(y_pred, y_true)
    print(f"Case 1: Partial correctness\nExpected: {expected}, Actual: {actual}")

    y_pred = [1, 0, 1, 1]
    y_true = [1, 0, 1, 1]
    expected = 1.0
    actual = get_accuracy(y_pred, y_true)
    print(f"Case 2: Perfect prediction\nExpected: {expected}, Actual: {actual}")

    y_pred = [1, 1, 1, 1]
    y_true = [0, 0, 0, 0]
    expected = 0.0
    actual = get_accuracy(y_pred, y_true)
    print(f"Case 3: Completely wrong prediction\nExpected: {expected}, Actual: {actual}")

    print("\n")


def test_get_precision():
    print("Testing get_precision...\n")

    y_pred = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    expected = 0.75
    actual = get_precision(y_pred, y_true, 1)
    print(f"Case 1: Mixed precision\nExpected: {expected}, Actual: {actual}")

    y_pred = [1, 1, 1, 1]
    y_true = [1, 0, 0, 1]
    expected = 0.5
    actual = get_precision(y_pred, y_true, 1)
    print(f"Case 2: Half correct predictions\nExpected: {expected}, Actual: {actual}")

    y_pred = [0, 0, 0, 0]
    y_true = [1, 1, 1, 1]
    expected = 0.0
    actual = get_precision(y_pred, y_true, 1)
    print(f"Case 3: No correct positive predictions\nExpected: {expected}, Actual: {actual}")

    print("\n")


def test_get_recall():
    print("Testing get_recall...\n")

    y_pred = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    expected = 0.6
    actual = get_recall(y_pred, y_true, 1)
    print(f"Case 1: Partial recall\nExpected: {expected}, Actual: {actual}")

    y_pred = [1, 1, 1, 1]
    y_true = [1, 0, 0, 1]
    expected = 1.0
    actual = get_recall(y_pred, y_true, 1)
    print(f"Case 2: Perfect recall\nExpected: {expected}, Actual: {actual}")

    y_pred = [0, 0, 0, 0]
    y_true = [1, 1, 1, 1]
    expected = 0.0
    actual = get_recall(y_pred, y_true, 1)
    print(f"Case 3: No positive predictions, recall should be zero\nExpected: {expected}, Actual: {actual}")

    print("\n")


def test_get_fscore():
    print("Testing get_fscore...\n")

    y_pred = [1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    expected = 0.6667
    actual = round(get_fscore(y_pred, y_true, 1), 4)
    print(f"Case 1: Mixed precision and recall\nExpected: {expected}, Actual: {actual}")

    y_pred = [1, 1, 1, 1]
    y_true = [1, 0, 0, 1]
    expected = 0.6667
    actual = round(get_fscore(y_pred, y_true, 1), 4)
    print(f"Case 2: Balanced precision and recall\nExpected: {expected}, Actual: {actual}")

    y_pred = [0, 0, 0, 0]
    y_true = [1, 1, 1, 1]
    expected = 0.0
    actual = get_fscore(y_pred, y_true, 1)
    print(f"Case 3: No positives, f-score should be zero\nExpected: {expected}, Actual: {actual}")

    print("\n")

def run_all_tests():
    test_get_accuracy()
    test_get_fscore()
    test_get_precision()
    test_get_recall
    
if __name__ == "__main__":
    run_all_tests()