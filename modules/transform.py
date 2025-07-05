"""Transform module"""

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "Gender": 2,
}
NUMERICAL_FEATURES = ["Height", "Weight", "BMI", "Age"]
LABEL_KEY = "Label"


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def convert_num_to_one_hot(label_tensor, num_labels=2):
    """
    Convert a label (0 or 1) into a one-hot vector
    Args:
        int: label_tensor (0 or 1)
    Returns
        label tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """

    outputs = {}

    # KATEGORIKAL
    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(inputs[key], top_k=dim + 1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    # NUMERIKAL
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tf.expand_dims(
            tft.scale_to_0_1(inputs[feature]), -1
        )

    # LABEL
    outputs[transformed_name(LABEL_KEY)] = tft.compute_and_apply_vocabulary(
        inputs[LABEL_KEY], 4
    )

    return outputs
