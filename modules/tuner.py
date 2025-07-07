# Import library
import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

from transform import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, transformed_name
from trainer import input_fn


def model_builder(hp):
    """
    This function defines a Keras model and returns the model as a Keras object.
    """

    input_features = []

    for key in CATEGORICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(CATEGORICAL_FEATURES[key] + 1,), name=transformed_name(key))
        )

    for key in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(key))
        )

    concat = tf.keras.layers.concatenate(input_features)

    x = tf.keras.layers.Dense(
        hp.Choice('units_1', [128, 256, 512]),
        activation='relu'
    )(concat)
    x = tf.keras.layers.Dropout(hp.Choice('dropout_1', [0.2, 0.3, 0.4]))(x)

    x = tf.keras.layers.Dense(
        hp.Choice('units_2', [64, 128]),
        activation='relu'
    )(x)
    x = tf.keras.layers.Dropout(hp.Choice('dropout_2', [0.2, 0.3, 0.4]))(x)

    x = tf.keras.layers.Dense(
        hp.Choice('units_3', [32, 64]),
        activation='relu'
    )(x)
    x = tf.keras.layers.Dropout(hp.Choice('dropout_3', [0.2, 0.3, 0.4]))(x)

    num_classes = 4  # Ganti jika jumlah kelas berbeda
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model


def tuner_fn(fn_args: FnArgs):
    """
    Build the tuner and return TunerFnResult to TFX.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(
        fn_args.train_files,
        tf_transform_output,
        batch_size=32
    )

    eval_dataset = input_fn(
        fn_args.eval_files,
        tf_transform_output,
        batch_size=32
    )

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_sparse_categorical_accuracy',
        max_trials=10,
        directory=fn_args.working_dir,
        project_name='tfx_tuner'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 10,
            "callbacks": [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
        }
    )

