from utils.model.layers import *
from utils.model.losses import *
from utils.model.model import get_optimizer
from vision.model.layers import SplitPathwaysVision
from utils.utils import *
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

from vision.model.losses import ContrastiveSoftmaxLoss, LateralPredictiveLoss

PATHWAY_TO_CLS = None


def get_data_augmentation(image_size, normalization_kwargs={}, rotation_factor=0.02, random_zoom_factor=0.2):
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(**normalization_kwargs),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=rotation_factor),
            layers.RandomZoom(random_zoom_factor),
        ],
        name="data_augmentation",
    )
    return data_augmentation


def create_model(name='model', koleo_lambda=0, classifier=False, l2=False,
                 input_shape=(32, 32, 3), num_classes=10, kernel_regularizer=None,
                 projection_dim=64, encoder='ViTEncoder', encoder_per_path=False,
                 image_size=72, patch_size=8, pathway_classification=True, pathway_classification_allpaths=False,
                 ensemble_classification=False, classifier_pathways=True,
                 augmentation_kwargs={}, encoder_kwargs={}, pathways_kwargs={},
                 predictive_embedding=None, predictive_embedding_kwargs={}, tokenizer_conv_kwargs=None):
    if isinstance(kernel_regularizer, str) and kernel_regularizer.startswith("tf."):
        kernel_regularizer = eval(kernel_regularizer)


    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = get_data_augmentation(image_size, **augmentation_kwargs)(inputs)

    if len(augmented.shape) == 3:
        augmented = augmented[..., None]

    if tokenizer_conv_kwargs is not None:
        augmented = ConvNet(channels=projection_dim, **tokenizer_conv_kwargs)(augmented)

    # Create patches.
    patches = Patches(patch_size, name=name + '_patch')(augmented)
    num_patches = (image_size // patch_size) ** 2

    # Encode patches.
    num_class_tokens = pathways_kwargs.get('n', 2) if pathways_kwargs.get('token_per_path', False) else (max(eval(pathways_kwargs.get('pathway_to_cls', '[0]'))) + 1)
    encoded_patches = PatchEncoder(num_patches, projection_dim, name=name + '_patchenc',
                                   kernel_regularizer=kernel_regularizer,
                                   num_class_tokens=num_class_tokens)(patches)

    # divide to different pathways
    if classifier and not classifier_pathways:
        pathways = [encoded_patches]
    else:
        pathways = SplitPathwaysVision(num_patches, name=name + '_pathways', **pathways_kwargs)(encoded_patches)
        pathways = [tf.squeeze(path, axis=-2) for path in tf.split(pathways, pathways.shape[-2], axis=-2)]

    import utils.model.encoders
    Encoder = get_class(encoder, utils.model.encoders)
    out_reg = KoLeoRegularizer(koleo_lambda) if koleo_lambda else (tf.keras.regularizers.L2(l2) if l2 else None)
    enc_init = lambda i: Encoder(name=name+f'_enc{i if i is not None else ""}',
                                 kernel_regularizer=kernel_regularizer, out_regularizer=out_reg, **encoder_kwargs)
    encoders = [enc_init(i) for i in range(len(pathways))] if encoder_per_path else [enc_init(None)] * len(pathways)

    embedding = tf.keras.layers.Concatenate(name=name + '_embedding', axis=-1)([encoder(pathway)[..., None]
                                                                                for encoder, pathway in zip(encoders, pathways)])

    outputs = [embedding]

    if predictive_embedding is not None:
        outputs.append(PredictiveEmbedding(predictive_embedding, name=name + "_predembd",
                                           dim=embedding.shape[1],
                                           regularization=predictive_embedding_kwargs.pop("regularization", kernel_regularizer),
                                           **predictive_embedding_kwargs)(embedding))

    # classification heads, with stop_grad unless classifier=True
    if pathway_classification:
        # Only for the first pathway
        embedding_for_classification = embedding if classifier else tf.stop_gradient(embedding)
        pathways_to_classify = list(range(len(pathways))) if pathway_classification_allpaths else [0]
        for path in pathways_to_classify:
            cur_embd = embedding_for_classification[..., path]
            cur_name = name + '_logits' + (str(path) if pathway_classification_allpaths else '')
            pathway_logits = layers.Dense(num_classes, activation=None,
                                          kernel_regularizer=kernel_regularizer, name=cur_name)(cur_embd)
            outputs.append(pathway_logits)
    if ensemble_classification:
        outputs.append(layers.Dense(num_classes, activation=None, kernel_regularizer=kernel_regularizer, name=name + '_ensemble_logits')(
            tf.reshape(embedding if classifier else tf.stop_gradient(embedding), (-1, np.multiply.reduce(embedding.shape[1:])))))

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_model(model, loss=ContrastiveSoftmaxLoss, loss_kwargs={},
                  optimizer_cls=tf.optimizers.legacy.Nadam if tf.__version__ == '2.12.0' else tf.optimizers.Nadam,
                  optimizer_kwargs={}, classifier=False, pathway_classification=True,
                  ensemble_classification=False, pathway_classification_allpaths=False,
                  metrics_kwargs={}, **kwargs):
    if kwargs:
        print(f"WARNING: compile_model got spare kwargs that won't be used: {kwargs}")

    losses = {}
    metrics = {}
    if classifier:
        losses[model.name + '_embedding'] = NullLoss()
    else:
        losses[model.name + '_embedding'] = loss(**loss_kwargs)

    if (model.name + "_predembd") in [l.name for l in model.layers]:
        losses[model.name + "_predembd"] = LateralPredictiveLoss(graph=model.get_layer(model.name + "_predembd").pred_graph)

    if metrics_kwargs:
        import utils.model.metrics as metrics_file
        metrics[model.name + "_embedding"] = get_class(metrics_kwargs['name'], metrics_file)(metrics_kwargs.get('kwargs', {}))

    if pathway_classification:
        if pathway_classification_allpaths:
            for path in range(model.get_layer(model.name + "_pathways").n):
                losses[model.name + f'_logits{path}'] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                metrics[model.name + f'_logits{path}'] = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        else:
            losses[model.name + '_logits'] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics[model.name + '_logits'] = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    if ensemble_classification:
        losses[model.name + '_ensemble_logits'] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics[model.name + '_ensemble_logits'] = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    for loss in losses.values():
        if hasattr(loss, "monitor") and loss.monitor is not None:
            if 'embedding' not in metrics:
                metrics['embedding'] = []
            elif not isinstance(metrics['embedding'], list):
                metrics['embedding'] = [metrics['embedding']]
            for k, m in loss.monitor.monitors.items():
                m.name = model.name + "_" + m.name
                metrics['embedding'].append(m)

    optimizer = get_optimizer(optimizer_cls=optimizer_cls, **optimizer_kwargs)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
