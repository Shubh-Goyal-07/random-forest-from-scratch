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
    return  impurity(parent_labels) - (ll*impurity(left_labels) + rl*impurity(right_labels))/pl

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    curr_best = {
        "feature_index": None,
        "threshold": None,
        "score": 0
    }

    for idx in feature_indices:
        feat_vals = features[:, idx]
        values = np.sort(np.unique(feat_vals))
        thresholds = (values[:-1] + values[1:]) / 2
        
        for thresh in thresholds:
            lf, ll, rf, rl = split_dataset(features, labels, idx, thresh)
            if len(ll) == 0 or len(rl) == 0:
                continue
            
            curr_score = split_score(labels, ll, rl)

            if curr_score>curr_best["score"]:
                curr_best["score"] = curr_score
                curr_best["feature_index"] = idx
                curr_best["threshold"] = thresh
    
    return curr_best

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    if len(labels)<min_samples_split or depth>=max_depth or len(np.unique(labels))==1:
        return True
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    values, counts = np.unique(labels, return_counts=True)
    mode = values[np.argmax(counts)]

    return int(mode)

# Step 7 - build_tree
import numpy as np

def build_tree(features,
               labels,
               max_depth=10,
               min_samples_split=2,
               feature_subset=None,
               depth=0):

    if feature_subset is None:
        feature_subset = [i for i in range(0, features.shape[1])]

    def process_node(features, labels, depth):

        if should_stop(labels,
                       depth,
                       max_depth,
                       min_samples_split):

            return {
                "leaf": True,
                "prediction": leaf_prediction(labels)
            }

        best_dict = best_split(
            features,
            labels,
            feature_subset
        )

        feature_index = best_dict["feature_index"]
        threshold = best_dict["threshold"]

        lf, ll, rf, rl = split_dataset(
            features,
            labels,
            feature_index,
            threshold
        )

        return {
            "leaf": False,
            "feature_index": feature_index,
            "threshold": threshold,
            "left": process_node(
                lf,
                ll,
                depth + 1
            ),
            "right": process_node(
                rf,
                rl,
                depth + 1
            )
        }

    return process_node(features, labels, depth)

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    if tree["leaf"]:
        return tree["prediction"]
    else:
        if example[tree["feature_index"]] <= tree["threshold"]:
            return predict_example_tree(tree["left"], example)
        else:
            return predict_example_tree(tree["right"], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    pred = []
    for row in features:
        pred.append(predict_example_tree(tree, row))
    
    return np.array(pred)

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    indices = rng.integers(0, len(features), size=len(features))

    return features[indices], labels[indices]

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    return rng.choice(num_features, size=num_to_pick, replace=False)

# Step 12 - train_forest
import numpy as np

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    np.random.seed(random_state)
    rng = np.random.default_rng(random_state)

    n_features = features.shape[1]

    if num_features_per_split is None:
        num_features_per_split = max(1, int(round(np.sqrt(n_features))))

    forest = []

    for _ in range(num_trees):

        # Bootstrap rows
        boot_features, boot_labels = bootstrap_sample(features, labels, rng)

        # Random subset of columns
        feature_indices = feature_subset(
            n_features,
            num_features_per_split,
            rng
        )

        # Train using only those columns
        tree = build_tree(
            boot_features,
            boot_labels,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            feature_subset=feature_indices
        )

        forest.append({
            "tree": tree,
            "feature_indices": feature_indices
        })

    return forest

# Step 13 - combine_predictions
from scipy import stats

def combine_predictions(tree_predictions):
    result = stats.mode(tree_predictions, axis=0)
    return result.mode

# Step 14 - predict_forest
def predict_forest(forest, features):
    preds = np.empty((len(features), len(forest)))
    for j, tree_info in enumerate(forest):
        tree_feats = features[:, tree_info["feature_indices"]]
        preds[:, j] = predict_tree(tree_info["tree"], tree_feats)
        
    return combine_predictions(preds)

# Step 15 - accuracy (not yet solved)
# TODO: implement

