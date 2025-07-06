import tensorflow as tf
import keras_tuner
from transform import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, transformed_name


def model_builder(hp):
    input_features = []

    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    concat = tf.keras.layers.concatenate(input_features)

    # Hyperparameter: number of units in first dense layer
    hp_units_1 = hp.Int("units_1", min_value=64, max_value=512, step=64)
    hp_units_2 = hp.Int("units_2", min_value=32, max_value=256, step=32)
    hp_units_3 = hp.Int("units_3", min_value=8, max_value=64, step=8)

    dense = tf.keras.layers.Dense(units=hp_units_1, activation="relu")(concat)
    dense = tf.keras.layers.Dense(units=hp_units_2, activation="relu")(dense)
    dense = tf.keras.layers.Dense(units=hp_units_3, activation="relu")(dense)

    num_classes = 4
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(dense)

    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    hp_learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def tuner_fn(fn_args):
    """Builds a Keras Tuner instance for TFX Tuner component."""

    tuner = keras_tuner.RandomSearch(
        model_builder,
        objective=keras_tuner.Objective(
            "val_sparse_categorical_accuracy", direction="max"
        ),
        max_trials=20,
        directory=fn_args.working_dir,
        project_name="tuner",
        overwrite=True,
    )

    return tuner
