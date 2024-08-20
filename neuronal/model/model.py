import tensorflow as tf
from tensorflow.keras import layers

from neuronal.model.losses import CrossPathwayTemporalContrastiveLoss
from neuronal.utils.data import Labels, CATEGORICAL
from utils.model.layers import SplitPathways
from utils.model.losses import NullLoss
from utils.model.model import get_optimizer
from utils.utils import get_class
import numpy as np


def get_neuronal_data_augmentation(bins_per_frame, **kwargs):
    data_augmentation = tf.keras.Sequential(
        [
            layers.Normalization(),
        ],
        name="data_augmentation",
    )
    return data_augmentation


class SplitPathwaysNeuronal(SplitPathways):
    """
    Call
    :param inputs: (B, S, T)
    :return: (B, d*S, N, DIM)
    """

    def __init__(self, num_units, n=2, d=0.5, intersection=True, fixed=False, seed=0, **kwargs):
        if isinstance(num_units, dict):
            raise NotImplementedError()
        super(SplitPathwaysNeuronal, self).__init__(num_signals=num_units, n=n, d=d, intersection=intersection,
                                                    fixed=fixed, seed=seed, class_token=False, **kwargs)
        self.num_units = num_units


def create_model(input_shape, name='neuronal_model', bins_per_frame=1,
                 classifier=False, l2=False, kernel_regularizer=None,
                 encoder='BasicRNN', encoder_per_path=False,
                 pathway_classification=True, pathway_classification_allpaths=False,
                 ensemble_classification=True, classifier_pathways=True,
                 augmentation_kwargs={}, encoder_kwargs={}, pathways_kwargs={}):
    if isinstance(kernel_regularizer, str) and kernel_regularizer.startswith("tf."):
        kernel_regularizer = eval(kernel_regularizer)

    inputs = layers.Input(shape=input_shape)
    if isinstance(input_shape, dict):
        units = {area: units for area, (units, bins_per_sample) in input_shape.items()}
        bins_per_sample = list(input_shape.values())[0][-1]
    else:
        units, bins_per_sample = input_shape

    frames = int(bins_per_sample / bins_per_frame)

    # Augment data.
    augmented = get_neuronal_data_augmentation(bins_per_frame, **augmentation_kwargs)(inputs)

    # divide to different pathways
    if classifier and not classifier_pathways:
        pathways = [augmented]
    else:
        pathways = SplitPathwaysNeuronal(units, name='pathways', **pathways_kwargs)(augmented)  # (B, d*S, N, T)
        pathways = tf.unstack(pathways, axis=-2)  # List[(B, d*S, T)]

    import utils.model.encoders
    Encoder = get_class(encoder, utils.model.encoders)
    out_reg = tf.keras.regularizers.L2(l2) if l2 else None
    enc_init = lambda i: Encoder(name=f'enc{i if i is not None else ""}',
                                 kernel_regularizer=kernel_regularizer,
                                 out_regularizer=out_reg, **encoder_kwargs)
    encoders = [enc_init(i) for i in range(len(pathways))] if encoder_per_path else [enc_init(None)] * len(pathways)

    embedding = tf.keras.layers.Concatenate(name='embedding', axis=-1)([encoder(pathway)[..., None]
                                                                                for encoder, pathway in
                                                                                zip(encoders, pathways)])
    # (B, P, DIM, T)

    outputs = [embedding]

    last_step_embedding = embedding[..., -bins_per_frame:]
    embedding_for_classification = last_step_embedding if classifier else tf.stop_gradient(last_step_embedding)
    embedding_for_classification = tf.reshape(embedding_for_classification,
                                              (tf.shape(embedding_for_classification)[0], len(pathways), embedding.shape[-2] * bins_per_frame))    # (B, P, DIMS*bins_per_frame)
    path_divide_embedding = tf.unstack(embedding_for_classification, axis=1)

    # classification heads, with stop_grad unless classifier=True
    if pathway_classification:
        # Only for the first pathway

        pathways_to_classify = list(range(len(pathways))) if pathway_classification_allpaths else [0]
        for path in pathways_to_classify:
            cur_embd = path_divide_embedding[path]
            for label in Labels:
                cur_name = 'logits' + (
                    str(path) if pathway_classification_allpaths else '') + f'_{label.value.name}'
                pathway_logits = layers.Dense(label.value.dimension, activation=None,
                                              kernel_regularizer=kernel_regularizer, name=cur_name)(cur_embd)
                outputs.append(pathway_logits)
    if ensemble_classification:
        for label in Labels:
            ens_inp = tf.concat(path_divide_embedding, axis=-1)
            ens_pred = layers.Dense(label.value.dimension, activation=None,
                                    kernel_regularizer=kernel_regularizer,
                                    name=f'ensemble_logits_{label.value.name}')(ens_inp)
            outputs.append(ens_pred)

    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_model(model, dataset, loss=CrossPathwayTemporalContrastiveLoss, loss_kwargs={},
                  optimizer_cls=tf.optimizers.legacy.Nadam if tf.__version__ == '2.12.0' else tf.optimizers.Nadam,
                  optimizer_kwargs={}, classifier=False, pathway_classification=True,
                  ensemble_classification=False, pathway_classification_allpaths=False,
                  metrics_kwargs={}, **kwargs):
    if kwargs:
        print(f"WARNING: compile_model got spare kwargs that won't be used: {kwargs}")

    losses = {}
    metrics = {}
    if classifier:
        losses['embedding'] = NullLoss()
    else:
        losses['embedding'] = loss(**loss_kwargs)
    dataset.update_name_to_label('embedding', Labels.STIMULUS)    # The label is irrelevant here

    if metrics_kwargs:
        import utils.model.metrics as metrics_file
        metrics["embedding"] = get_class(metrics_kwargs['name'], metrics_file)(
            metrics_kwargs.get('kwargs', {}))

    label_class_loss = {
        label.value.name: tf.keras.losses.SparseCategoricalCrossentropy if label.value.kind == CATEGORICAL else tf.keras.losses.MeanAbsoluteError
        for label in Labels}
    label_class_metric = {
        label.value.name: tf.keras.metrics.SparseCategoricalAccuracy if label.value.kind == CATEGORICAL else tf.keras.losses.MeanAbsoluteError
        for label in Labels}

    if pathway_classification:
        if pathway_classification_allpaths:
            for path in range(model.get_layer(model.name + "_pathways").n):
                for label in Labels:
                    dataset.update_name_to_label(f'logits{path}_{label.value.name}', label)
                    losses[f'logits{path}_{label.value.name}'] = label_class_loss[label.value.name]()
                    metrics[f'logits{path}_{label.value.name}'] = label_class_metric[label.value.name]()
        else:
            for label in Labels:
                dataset.update_name_to_label(f'logits_{label.value.name}', label)
                losses[f'logits_{label.value.name}'] = label_class_loss[label.value.name]()
                metrics[f'logits_{label.value.name}'] = label_class_metric[label.value.name]()

    if ensemble_classification:
        for label in Labels:
            dataset.update_name_to_label(f'_ensemble_logits_{label.value.name}', label)
            losses[f'ensemble_logits_{label.value.name}'] = label_class_loss[label.value.name]()
            metrics[f'ensemble_logits_{label.value.name}'] = label_class_metric[label.value.name]()

    optimizer = get_optimizer(optimizer_cls=optimizer_cls, **optimizer_kwargs)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
