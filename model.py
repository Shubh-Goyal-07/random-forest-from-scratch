"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    if len(labels) <= 1:
        return 0

    unique, counts = np.unique(labels, return_counts=True)
    probs = counts/len(labels)
    
    return -np.sum(probs*np.log(probs))

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    left_mask = features[:, feature_index] <= threshold
    right_mask = ~left_mask

    return (
        features[left_mask],
        labels[left_mask],
        features[right_mask],
        labels[right_mask]
    )

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    ll, rl, pl = len(left_labels), len(right_labels), len(parent_labels)
    return  impurity(parent_labels) - (ll*impurity(left_labels) + rl*impurity(right_labels))

# Step 4 - best_split (not yet solved)
# TODO: implement

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

