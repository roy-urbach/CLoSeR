import tensorflow as tf
from tensorflow.keras import layers

from neuronal.model.losses import TemporalContiguityLoss
from neuronal.utils.data import Labels
from utils.model.layers import SplitPathways, Stack
from utils.model.losses import NullLoss
from utils.model.optimizer import get_optimizer
from utils.modules import Modules
from utils.utils import get_class


def get_neuronal_data_augmentation(augmentations=[], name='data_augmentation', axis=-1, **kwargs):
    data_augmentation = tf.keras.Sequential(
        [
            layers.Normalization(axis=axis),
        ] + augmentations,
        name=name,
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


def create_model(input_shape, name='neuronal_model',
                 classifier=False, l2=False, kernel_regularizer=None, kernel_regularizer_sup=None,
                 encoder='TimeAgnosticMLP', encoder_per_path=False,
                 pathway_classification=True, pathway_classification_allpaths=False,
                 ensemble_classification=True, classifier_pathways=True, bins_per_frame=1,
                 augmentation_kwargs={}, encoder_kwargs={}, pathways_kwargs={}, labels=Labels, label_to_dim=None,
                 module=Modules.NEURONAL, SplitClass=SplitPathwaysNeuronal):
    """
    a neuronal model creating function

    :param input_shape: shape of the input, or a dictionary with {area: shape}
    :param name: name of the model
    :param classifier: whether it is supervised or not
    :param l2: l2 regularization on the embedding
    :param kernel_regularizer: kernel regularization for the encoder
    :param kernel_regularizer_sup: kernel regularization for the supervised head
    :param encoder: class of the encoder
    :param encoder_per_path: whether to have individual encoders or shared weights
    :param pathway_classification: whether the classification is done for independent classifiers (even when it is not a classifier, it is usefull to have this metric during training)
    :param pathway_classification_allpaths: if pathways classification, whether each encoder should be trained in a supervised manner, or only the first
    :param ensemble_classification: whether a linear classifier from the concatenated output of all encoders should learn to classify
    :param classifier_pathways: if false, the classifier receives the full image and is a single encoder
    :param augmentation_kwargs: kwargs for get_data_augmentation
    :param encoder_kwargs: kwargs for the encoder
    :param pathways_kwargs: kwargs for SplitPathwaysVision
    :param labels: labels to train the supervised heads with
    :param label_to_dim: a dictionary with {label: dimension}
    :param module: the module
    :param SplitClass: class of the splitting object
    :return: a tensorflow Model
    """

    if isinstance(kernel_regularizer, str) and kernel_regularizer.startswith("tf."):
        kernel_regularizer = eval(kernel_regularizer)

    if isinstance(input_shape, dict):
        different_areas = True
        areas = sorted(list(input_shape.keys()))
        units = {area: units for area, (units, bins_per_sample) in input_shape.items()}
        inputs = [layers.Input(shape=input_shape[area], name=area) for area in areas]  # [(B, N, T)]
    else:
        different_areas = False
        units, bins_per_sample = input_shape
        inputs = layers.Input(shape=input_shape, name='inp')  # (B, N, T)

    # Augment data.
    if different_areas:
        augmented = [get_neuronal_data_augmentation(**augmentation_kwargs, name=f'data_augmentation_{area}')(inp)
                     for area, inp in zip(areas, inputs)]
    else:
        augmented = get_neuronal_data_augmentation(**augmentation_kwargs)(inputs)

    # divide to different pathways
    if classifier and not classifier_pathways:
        pathways = [augmented]
    elif different_areas:
        pathways = augmented
    else:
        pathways = SplitClass(units, name='pathways', **pathways_kwargs)(augmented)  # (B, d*S, N, T)
        pathways = tf.unstack(pathways, axis=-2)  # List[(B, d*S, T)]


    import utils.model.encoders
    Encoder = get_class(encoder, utils.model.encoders)
    out_reg = tf.keras.regularizers.L2(l2) if l2 else None

    if encoder_kwargs.get("local", False):
        encoder_kwargs['loss'] = module.get_loss(encoder_kwargs.pop("loss"))(**encoder_kwargs.pop("loss_kwargs", {}))

    enc_init = lambda i: Encoder(name=f'enc{i if i is not None else ""}',
                                 kernel_regularizer=kernel_regularizer,
                                 out_regularizer=out_reg, **encoder_kwargs)
    encoders = [enc_init(i if not different_areas else areas[i])
                for i in range(len(pathways))] if encoder_per_path else [enc_init(None)] * len(pathways)

    embedding = Stack(name='embedding', axis=-1)(*[encoder(pathway) for encoder, pathway in zip(encoders, pathways)])
    # (B, T, DIM, P)
    outputs = [embedding]

    # Supervised readout part (with stopgrad if classifier=False):
    last_step_embedding = embedding[:, -1]
    last_step_embedding = tf.reshape(last_step_embedding,     # (B, DIMS*bins_per_frame, P)
                                     (tf.shape(last_step_embedding)[0],
                                      embedding.shape[-2],
                                      len(pathways)))
    embedding_for_classification = last_step_embedding if classifier else tf.stop_gradient(last_step_embedding)
    path_divide_embedding = tf.unstack(embedding_for_classification, axis=-1)   # List[(B, DIMS*bins_per_frame)]

    labels = [eval(label) if isinstance(label, str) else label for label in labels]

    # classification heads, with stop_grad unless classifier=True
    if pathway_classification:
        # Only for the first pathway

        pathways_to_classify = list(range(len(pathways))) if pathway_classification_allpaths else [0]
        for path in pathways_to_classify:
            cur_embd = path_divide_embedding[path]
            for label in labels:
                cur_name = 'logits' + (
                    str(path) if pathway_classification_allpaths else '') + f'_{label.value.name}'
                dim = label.value.dimension if label.value.dimension else input_shape[1]
                if label_to_dim is not None and label.value.name in label_to_dim:
                    dim = label_to_dim[label.value.name]
                pathway_logits = layers.Dense(dim, activation=None,
                                              kernel_regularizer=kernel_regularizer_sup if kernel_regularizer_sup else kernel_regularizer,
                                              name=cur_name)(cur_embd)
                outputs.append(pathway_logits)
    if ensemble_classification:
        for label in labels:
            ens_inp = tf.concat(path_divide_embedding, axis=-1)
            dim = label.value.dimension if label.value.dimension else input_shape[1]
            if label_to_dim is not None and label.value.name in label_to_dim:
                dim = label_to_dim[label.value.name]
            ens_pred = layers.Dense(dim,
                                    activation=None,
                                    kernel_regularizer=kernel_regularizer_sup if kernel_regularizer_sup else kernel_regularizer,
                                    name=f'ensemble_logits_{label.value.name}')(ens_inp)
            outputs.append(ens_pred)

    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def label_to_loss(label):
    if label.value.is_categorical():
        if label.value.dimension > 1:
            return tf.keras.losses.SparseCategoricalCrossentropy
        else:
            return tf.keras.losses.BinaryCrossentropy
    else:
        return tf.keras.losses.MeanAbsoluteError


def label_to_metric(label):
    if label.value.is_categorical():
        if label.value.dimension > 1:
            return tf.keras.metrics.SparseCategoricalAccuracy
        else:
            return tf.keras.metrics.BinaryAccuracy
    else:
        return tf.keras.losses.MeanAbsoluteError


def compile_model(model, dataset, loss=TemporalContiguityLoss, loss_kwargs={},
                  optimizer_cls=tf.optimizers.legacy.Nadam if tf.__version__ == '2.12.0' else tf.optimizers.Nadam,
                  optimizer_kwargs={}, classifier=False, pathway_classification=True,
                  ensemble_classification=True, pathway_classification_allpaths=False,
                  metrics_kwargs={}, labels=Labels, **kwargs):
    if kwargs:
        print(f"WARNING: compile_model got spare kwargs that won't be used: {kwargs}")

    losses = {}
    metrics = {}
    if classifier:
        losses['embedding'] = NullLoss()
    else:
        losses['embedding'] = loss(**loss_kwargs)
    dataset.update_name_to_label('embedding', Labels.NOTHING)    # The label is irrelevant here

    # supervised readouts losses
    labels = [eval(label) if isinstance(label, str) else label for label in labels]

    if metrics_kwargs:
        import utils.model.metrics as metrics_file
        metrics["embedding"] = get_class(metrics_kwargs['name'], metrics_file)(
            metrics_kwargs.get('kwargs', {}))

    label_kwargs = {label.value.name: dict(from_logits=True) if label.value.is_categorical() else {} for label in labels}

    P = model.get_layer("pathways").n if 'pathways' in [l.name for l in model.layers] else model.get_layer("embedding").output_shape[-1]
    if pathway_classification:
        if pathway_classification_allpaths:
            for path in range(P):
                for label in labels:
                    dataset.update_name_to_label(f'logits{path}_{label.value.name}', label)
                    losses[f'logits{path}_{label.value.name}'] = label_to_loss(label)(**label_kwargs[label.value.name])
                    metrics[f'logits{path}_{label.value.name}'] = label_to_metric(label)[label.value.name]()
        else:
            for label in labels:
                dataset.update_name_to_label(f'logits_{label.value.name}', label)
                losses[f'logits_{label.value.name}'] = label_to_loss(label)(**label_kwargs[label.value.name])
                metrics[f'logits_{label.value.name}'] = label_to_metric(label)[label.value.name]()

    if ensemble_classification:
        for label in labels:
            dataset.update_name_to_label(f'ensemble_logits_{label.value.name}', label)
            losses[f'ensemble_logits_{label.value.name}'] = label_to_loss(label)(**label_kwargs[label.value.name])
            metrics[f'ensemble_logits_{label.value.name}'] = label_to_metric(label)[label.value.name]()


    # added individual losses to metrics (using monitors)
    for loss in losses.values():
        if hasattr(loss, "monitor") and loss.monitor is not None:
            if 'embedding' not in metrics:
                metrics['embedding'] = []
            elif not isinstance(metrics['embedding'], list):
                metrics['embedding'] = [metrics['embedding']]
            for k, m in loss.monitor.monitors.items():
                metrics['embedding'].append(m)

    optimizer = get_optimizer(optimizer_cls=optimizer_cls, **optimizer_kwargs)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
