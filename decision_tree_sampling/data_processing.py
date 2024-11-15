from collections import Counter
from sklearn.tree import DecisionTreeClassifier

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

    return rule_counts

def get_rule_stats(X, y, rules):
  n = len(X)
  allFreqX = freqX(X, rules)
  allFreqY = freqY(y, rules, [116, 117])
  allFreqZ = freqZ(X, y, rules, [116, 117])

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
  
  return rule_stats
