import tensorflow as tf
from tf.keras import layers

from neuronal.model.losses import CrossPathwayTemporalContrastiveLoss, SparseCategoricalCrossEntropyByKey, \
    MeanAbsoluteErrorByKeyLoss
from neuronal.utils.data import Labels, CATEGORICAL
from utils.model.layers import SplitPathways
from utils.model.losses import NullLoss
from utils.model.metrics import SparseCategoricalAccuracyByKey, MeanAbsoluteErrorByKeyMetric
from utils.utils import get_class
from utils.tf_utils import serialize
import numpy as np


def get_neuronal_data_augmentation(inputs, **kwargs):
    # TODO: implement
    return inputs


class SplitPathwaysNeuronal(SplitPathways):
    """
    Call
    :param inputs: (B, S, T)
    :return: (B, d*S, N, DIM)
    """

    def __init__(self, num_units, n=2, d=0.5, intersection=True, fixed=False, seed=0, **kwargs):
        super(SplitPathwaysNeuronal, self).__init__(num_signals=num_units, n=n, d=d, intersection=intersection,
                                                    fixed=fixed, seed=seed, class_token=False, **kwargs)
        self.num_units = num_units


def create_neuronal_model(name='neuronal_model', frames=9, bins_per_frame=1, units=100,
                          classifier=False, l2=False, kernel_regularizer=None,
                          encoder='BasicRNN', encoder_per_path=False,
                          pathway_classification=True, pathway_classification_allpaths=False,
                          ensemble_classification=False, classifier_pathways=True,
                          augmentation_kwargs={}, encoder_kwargs={}, pathways_kwargs={}):
    if isinstance(kernel_regularizer, str) and kernel_regularizer.startswith("tf."):
        kernel_regularizer = eval(kernel_regularizer)

    inputs = layers.Input(shape=(frames * bins_per_frame, units))
    # Augment data.
    augmented = get_neuronal_data_augmentation(bins_per_frame, **augmentation_kwargs)(inputs)

    # divide to different pathways
    if classifier and not classifier_pathways:
        pathways = [augmented]
    else:
        pathways = SplitPathwaysNeuronal(units, name=name + '_pathways', **pathways_kwargs)(augmented)  # (B, d*S, N, T)
        pathways = [tf.squeeze(path, axis=-2) for path in tf.split(pathways, pathways.shape[-2], axis=-2)] # List[(B, d*S, T)]

    import utils.model.encoders
    Encoder = get_class(encoder, utils.model.encoders)
    out_reg = tf.keras.regularizers.L2(l2) if l2 else None
    enc_init = lambda i: Encoder(name=name+f'_enc{i if i is not None else ""}',
                                 kernel_regularizer=kernel_regularizer, out_regularizer=out_reg, **encoder_kwargs)
    encoders = [enc_init(i) for i in range(len(pathways))] if encoder_per_path else [enc_init(None)] * len(pathways)

    embedding = tf.keras.layers.Concatenate(name=name + '_embedding', axis=-1)([encoder(pathway)[..., None]
                                                                                for encoder, pathway in zip(encoders, pathways)])
    # (B, T, DIM, P)

    outputs = [embedding]

    last_step_embedding = embedding[:, -1]

    # classification heads, with stop_grad unless classifier=True
    if pathway_classification:
        # Only for the first pathway
        embedding_for_classification = last_step_embedding if classifier else tf.stop_gradient(last_step_embedding)
        if bins_per_frame > 1:
            embedding_for_classification = embedding_for_classification[:, bins_per_frame - 1::bins_per_frame]

        pathways_to_classify = list(range(len(pathways))) if pathway_classification_allpaths else [0]
        for path in pathways_to_classify:
            cur_embd = embedding_for_classification[..., path]
            for label in Labels:
                cur_name = name + '_logits' + (str(path) if pathway_classification_allpaths else '') + f'_{label.value.name}'
                pathway_logits = layers.Dense(label.value.dimension, activation=None,
                                              kernel_regularizer=kernel_regularizer, name=cur_name)(cur_embd)
                outputs.append(pathway_logits)
    if ensemble_classification:
        for label in Labels:
            outputs.append(layers.Dense(label.value.dimension, activation=None,
                                        kernel_regularizer=kernel_regularizer, name=name + f'_ensemble_logits_{label.value.name}')(
                tf.reshape(embedding if classifier else tf.stop_gradient(embedding), (-1, np.multiply.reduce(embedding.shape[1:])))))

    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_model(model, loss=CrossPathwayTemporalContrastiveLoss, loss_kwargs={},
                  optimizer_cls=tf.optimizers.legacy.Nadam if tf.__version__ == '2.12.0' else tf.optimizers.Nadam,
                  optimizer_kwargs={}, classifier=False, pathway_classification=True,
                  ensemble_classification=False, pathway_classification_allpaths=False,
                  metrics_kwargs={}, **kwargs):
    if kwargs:
        print(f"WARNING: compile_model got spare kwargs that won't be used: {kwargs}")

    if "optimizer" in optimizer_kwargs:
        optimizer_cls_name = optimizer_kwargs.pop("optimizer")
        try:
            optimizer_cls = eval(optimizer_cls_name)
        except Exception as err:
            print(f"Tried to eval {optimizer_cls_name}, get error:", err)
            try:
                optimizer_cls = tf.keras.optimizers.getattr(optimizer_cls_name)
            except Exception as err:
                print(f"Tried to getattr tf.keras.optimizers.{optimizer_cls_name}, get error:", err)
    print(f"using optimizer {optimizer_cls}")
    optimizer = optimizer_cls(**{k: eval(v) if isinstance(v, str) and v.startswith("tf.") else v
                                 for k, v in optimizer_kwargs.items()})
    serialize(optimizer.__class__, 'Custom')

    losses = {}
    metrics = {}
    if classifier:
        losses[model.name + '_embedding'] = NullLoss()
    else:
        losses[model.name + '_embedding'] = loss(**loss_kwargs)

    if metrics_kwargs:
        import utils.model.metrics as metrics_file
        metrics[model.name + "_embedding"] = get_class(metrics_kwargs['name'], metrics_file)(metrics_kwargs.get('kwargs', {}))

    label_class_loss = {
        label.value.name: SparseCategoricalCrossEntropyByKey if label.value.kind == CATEGORICAL else MeanAbsoluteErrorByKeyLoss
        for label in Labels}
    label_class_metric = {
        label.value.name: SparseCategoricalAccuracyByKey if label.value.kind == CATEGORICAL else MeanAbsoluteErrorByKeyMetric
        for label in Labels}

    if pathway_classification:
        if pathway_classification_allpaths:
            for path in range(model.get_layer(model.name + "_pathways").n):
                for label in Labels:
                    losses[model.name + f'_logits{path}_{label.value.name}'] = label_class_loss[label.value.name](label.value.name, from_logits=True)
                    metrics[model.name + f'_logits{path}_{label.value.name}'] = label_class_metric[label.value.name](label.value.name)
        else:
            for label in Labels:
                losses[model.name + f'_logits_{label.value.name}'] = label_class_loss[label.value.name](label.value.name, from_logits=True)
                metrics[model.name + f'_logits_{label.value.name}'] = label_class_metric[label.value.name](label.value.name)

    if ensemble_classification:
        for label in Labels:
            losses[model.name + f'_ensemble_logits_{label.value.name}'] = label_class_loss[label.value.name](label.value.name, from_logits=True)
            metrics[model.name + f'_ensemble_logits_{label.value.name}'] = label_class_metric[label.value.name](label.value.name, name="accuracy")
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
