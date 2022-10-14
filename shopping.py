import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    # Read data from file
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)

        # Create empty tuple with one list for evidence and one for labels
        data = ([],[])
        # Reads every row in csv file
        for row in reader:
            evidence = []
            # Add int/float values at their corresponding positions in list
            evidence.insert(0, int(row[0]))
            evidence.insert(1, float(row[1]))
            evidence.insert(2, int(row[2]))
            evidence.insert(3, float(row[3]))
            evidence.insert(4, int(row[4]))
            evidence.insert(5, float(row[5]))
            evidence.insert(6, float(row[6]))
            evidence.insert(7, float(row[7]))
            evidence.insert(8, float(row[8]))
            evidence.insert(9, float(row[9]))

            # Create month dictionary to identify each month's #
            months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6,
                      "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
            evidence.insert(10, months[row[10]])

            # Add int/float values at their corresponding positions in list
            evidence.insert(11, int(row[11]))
            evidence.insert(12, int(row[12]))
            evidence.insert(13, int(row[13]))
            evidence.insert(14, int(row[14]))

            # Create visitor_type dictionary to identify corresponding #
            visitor_type = {"Returning_Visitor": 1, "New_Visitor": 0, "Other": 0}
            evidence.insert(15, visitor_type[row[15]])

            # Create true_false dictionary to identify each boolean value's #
            true_false = {"FALSE": 0, "TRUE": 1}
            evidence.insert(16, true_false[row[16]])

            # Add evidence list to first list in data
            data[0].append(evidence)

            # Add true/false value to second list in data
            data[1].append(true_false[row[17]])

    return data


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Create k nearest neighbor model with k = 1
    model = KNeighborsClassifier(n_neighbors=1)
    # Fit model
    model.fit(evidence, labels)
    return model

    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Variables to keep track of total # of actual positives and negatives
    positives = 0
    negatives = 0

    # Traverse through labels
    for label in labels:
        # If label is true, add to positive count
        if label == 1:
            positives += 1
        # If label is false, add to negative count
        elif label == 0:
            negatives += 1

    # Variables to keep track of correctly identified positives and negatives
    acc_pos = 0
    acc_neg = 0

    # Loop over the length of labels
    for val in range(len(labels)):
        # If the values in labels and predictions at position "val" are both 1, add to acc_pos
        if labels[val] == 1 and predictions[val] == 1:
            acc_pos += 1
        # If the values in labels and predictions at position "val" are both 0, add to acc_neg
        if labels[val] == 0 and predictions[val] == 0:
            acc_neg += 1

    # Return correctly identified positives/total positives and correctly identified negatives/total negatives
    return acc_pos/positives, acc_neg/negatives

    raise NotImplementedError


if __name__ == "__main__":
    main()
