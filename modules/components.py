"""Initiate tfx pipeline components"""

import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Tuner,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)


def init_components(
    data_dir,
    transform_module,
    training_module,
    tuner_module,
    training_steps,
    eval_steps,
    serving_model_dir,
):
    """Initiate tfx pipeline components

    Args:
        data_dir (str): a path to the data
        transform_module (str): a path to the transform_module
        training_module (str): a path to the transform_module
        training_steps (int): number of training steps
        eval_steps (int): number of eval steps
        serving_model_dir (str): a path to the serving model directory

    Returns:
        TFX components
    """
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )

    example_gen = CsvExampleGen(input_base=data_dir, output_config=output)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(transform_module),
    )

    tuner = Tuner(
        module_file=os.path.abspath(tuner_module),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps),
    )

    trainer = Trainer(
        module_file=os.path.abspath(training_module),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=training_steps),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps),
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

    slicing_specs = [
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=["Gender_xf"]),
    ]

    metrics_specs = [
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name="AUC"),
                tfma.MetricConfig(class_name="Precision"),
                tfma.MetricConfig(class_name="Recall"),
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(
                    class_name="SparseCategoricalAccuracy",
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": 0.0001},
                        ),
                    ),
                ),
            ]
        )
    ]

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Label_xf")],
        slicing_specs=slicing_specs,
        metrics_specs=metrics_specs,
    )

    evaluator = Evaluator(
        examples=transform.outputs["transformed_examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )

    return components
