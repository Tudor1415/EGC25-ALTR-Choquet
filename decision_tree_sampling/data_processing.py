from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extract_random_forest_rules(X, y, n_estimators=10, max_depth=5):
    """
    Fit a Random Forest classifier and extract all decision rules from all trees in the forest.

    Parameters:
    X (numpy.ndarray or pandas.DataFrame): Feature data
    y (numpy.ndarray or pandas.Series): Target data

    Returns:
    list: A list of decision rules from all trees. Each rule is a list of (feature, operator, threshold, class_label) tuples.
    """
    # Fit the Random Forest
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X, y)

    all_rules = []

    for tree_idx, tree in enumerate(clf.estimators_):
        # Extract the tree structure
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        value = tree.tree_.value

        # Helper function to traverse the tree and extract rules
        def extract_rules(node_id, current_path):
            if children_left[node_id] == children_right[node_id]:
                class_label = value[node_id].argmax()
                return [current_path + [(feature[node_id], '<=', threshold[node_id], class_label)]]

            # Internal node
            rules = []
            # Left child
            if children_left[node_id] != -1:
                left_path = current_path + [(feature[node_id], '<=', threshold[node_id], None)]
                rules.extend(extract_rules(children_left[node_id], left_path))
            # Right child
            if children_right[node_id] != -1:
                right_path = current_path + [(feature[node_id], '>', threshold[node_id], None)]
                rules.extend(extract_rules(children_right[node_id], right_path))
            return rules

        # Extract all decision rules from this tree
        tree_rules = extract_rules(0, [])
        all_rules.extend(tree_rules)

    return all_rules
def extract_decision_tree_rules(X, y, max_depth = 5):
    """
    Fit a decision tree classifier and extract all decision rules as paths from root to leaf, including class labels.

    Parameters:
    X (numpy.ndarray or pandas.DataFrame): Feature data
    y (numpy.ndarray or pandas.DataFrame): Target data

    Returns:
    list: A list of decision rules, where each rule is a list of (feature, operator, threshold, class_label) tuples.
    """
    # Fit the decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)

    # Extract the tree structure
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    value = clf.tree_.value

    # Helper function to traverse the tree and extract rules
    def extract_rules(node_id, current_path):
        if children_left[node_id] == children_right[node_id]:
            # Leaf node, return the current path and the class label
            class_label = value[node_id].argmax()
            return [current_path + [(feature[node_id], '<=', threshold[node_id], class_label)]]

        # Internal node
        rules = []
        if children_left[node_id] != -1:
            left_path = current_path + [(feature[node_id], '<=', threshold[node_id], None)]
            rules.extend(extract_rules(children_left[node_id], left_path))
        if children_right[node_id] != -1:
            right_path = current_path + [(feature[node_id], '>', threshold[node_id], None)]
            rules.extend(extract_rules(children_right[node_id], right_path))
        return rules

    # Extract all decision rules
    return extract_rules(0, [])

def freqX(X, rules):
    """
    Count the number of transactions in X that satisfy all conditions in each decision tree rule.

    Parameters:
    X (numpy.ndarray or pandas.DataFrame): Feature data
    rules (list): A list of decision rules, where each rule is a list of (feature, operator, threshold, class_label) tuples.

    Returns:
    dict: A dictionary where the keys are the rules and the values are the counts of transactions satisfying each rule.
    """
    rule_counts = {}

    for rule in rules:
        mask = True
        for feature, operator, threshold, _ in rule:
            feature = int(feature)
            threshold = float(threshold)
            if(feature < 0):
              break
            if operator == '<=':
                mask &= X[feature] <= threshold
            else:
                mask &= X[feature] > threshold
        rule_counts[tuple(rule)] = mask.sum()

    return rule_counts

def freqY(y, rules, class_values):
    """
    Count the number of times each class label appears in the target variable y.

    Parameters:
    y (numpy.ndarray or pandas.Series): Target data

    Returns:
    dict: A dictionary where the keys are the class labels and the values are the counts.
    """
    rule_counts = {}

    for rule in rules:
        mask = True
        for feature, operator, threshold, class_idx in rule:
          if class_idx is not None:
            mask &= y == class_values[class_idx]
            break

        rule_counts[tuple(rule)] = mask.sum()

    return rule_counts

def freqZ(X, y, rules, class_values):
    """
    Count the number of times each class label appears in the target variable y.

    Parameters:
    y (numpy.ndarray or pandas.Series): Target data

    Returns:
    dict: A dictionary where the keys are the class labels and the values are the counts.
    """
    rule_counts = {}

    for rule in rules:
        mask = True
        for feature, operator, threshold, class_idx in rule:
            feature = int(feature)
            threshold = float(threshold)
            if class_idx is not None:
              mask &= y == class_values[class_idx]
              break
            if operator == '<=':
                mask &= X[feature] <= threshold
            else:
                mask &= X[feature] > threshold
        rule_counts[tuple(rule)] = mask.sum()

    return rule_counts, mask

def get_rule_stats(X, y, rules, class_values):
  n = len(X)
  allFreqX = freqX(X, rules)
  allFreqY = freqY(y, rules, class_values)
  allFreqZ, cover = freqZ(X, y, rules, class_values)

  rule_stats = {}
  for rule in rules:
      rule_stats[tuple(rule)] = {}
      n1x = allFreqX[tuple(rule)]
      rule_stats[tuple(rule)]['n1x'] = n1x 
      nx1 = allFreqY[tuple(rule)]
      rule_stats[tuple(rule)]['nx1'] = nx1 
      n11 = allFreqZ[tuple(rule)]
      rule_stats[tuple(rule)]['n11'] = n11 
      rule_stats[tuple(rule)]['nx0'] = n - nx1
      n0x = n - n1x
      rule_stats[tuple(rule)]['n0x'] = n0x 
      rule_stats[tuple(rule)]['n10'] = n1x - n11
      n01 = nx1 - n11
      rule_stats[tuple(rule)]['n01'] = n01 
      rule_stats[tuple(rule)]['n00'] = n0x - n01
      rule_stats[tuple(rule)]['cover'] = cover
  
  return rule_stats

def jaccard_distance(cover1, cover2):
    """
    Calculate the Jaccard distance between two covers.

    Args:
    cover1 (array-like): Boolean array indicating the coverage of the first rule.
    cover2 (array-like): Boolean array indicating the coverage of the second rule.

    Returns:
    float: Jaccard distance between the two covers.
    """
    # Ensure input is in numpy array format
    cover1 = np.asarray(cover1)
    cover2 = np.asarray(cover2)

    # Calculate the intersection and union
    intersection = np.logical_and(cover1, cover2).sum()
    union = np.logical_or(cover1, cover2).sum()

    # Handle case where union is zero to avoid division by zero
    if union == 0:
        return 1.0  # Maximum distance if both sets are empty

    # Compute Jaccard distance
    return 1 - intersection / union

def compute_jaccard_distance_matrix(covers):
    """
    Compute the Jaccard distance matrix for a set of rule covers.

    Args:
    covers (list of array-like): A list where each element is a boolean array indicating the cover of a rule.

    Returns:
    np.ndarray: A 2D Numpy array representing the Jaccard distance matrix.
    """
    n = len(covers)
    distance_matrix = np.zeros((n, n))  # Initialize a square matrix

    for i in range(n):
        for j in range(i + 1, n):  # Only compute the upper triangle
            distance = jaccard_distance(covers[i], covers[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # Mirror the distance since the matrix is symmetric

    return distance_matrix