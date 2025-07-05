import os
import tensorflow as tf
import tensorflow_transform as tft

from transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)


def get_model(show_summary=True):
    """Define Keras model."""
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
    deep = tf.keras.layers.Dense(256, activation="relu")(concat)
    deep = tf.keras.layers.Dense(64, activation="relu")(deep)
    deep = tf.keras.layers.Dense(16, activation="relu")(deep)

    num_classes = 4
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(deep)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    if show_summary:
        model.summary()

    return model


def gzip_reader_fn(filenames):
    """Load compressed data."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def get_serve_json_fn(model, tf_transform_output):
    """Return serving function that accepts JSON dict directly."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_json_fn(inputs):
        transformed_features = model.tft_layer(inputs)
        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_json_fn


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Input fn for training and eval."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    label_key = transformed_name(LABEL_KEY)

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=label_key,
        shuffle=True,
        drop_final_batch=True,
    )

    return dataset


def run_fn(fn_args):
    """Train the model."""

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=10,
    )

    signatures = {
        "serving_default": get_serve_json_fn(
            model, tf_transform_output
        ).get_concrete_function(
            {
                "Height": tf.TensorSpec([None, 1], dtype=tf.int64),
                "Weight": tf.TensorSpec([None, 1], dtype=tf.int64),
                "BMI": tf.TensorSpec([None, 1], dtype=tf.float32),
                "Age": tf.TensorSpec([None, 1], dtype=tf.int64),
                "Gender": tf.TensorSpec([None, 1], dtype=tf.string),
            }
        )
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
