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


def create_model(name='model', classifier=False, l2=False, divide_patches=True,
                 divide_rgb=False, input_shape=(32, 32, 3), num_classes=10, kernel_regularizer=None,
                 projection_dim=64, encoder='ViTEncoder', encoder_per_path=False,
                 image_size=72, patch_size=8, pathway_classification=True, pathway_classification_allpaths=False,
                 ensemble_classification=False, classifier_pathways=True,
                 augmentation_kwargs={}, encoder_kwargs={}, pathways_kwargs={},
                 patch_encoder=True, patch_encoder_after_split=False, label_to_dim=None):
    """
    a vision model creationg function
    :param name: name of the model
    :param classifier: whether it is supervised or not
    :param l2: l2 regularization on the embedding
    :param divide_patches: whether to divide the images to patches or not (ViT requires it, MLP doesn't)
    :param divide_rgb: if not divide_patches, whether to keep the RGB channels before splitting to different encoders
    :param input_shape: shape of the input
    :param num_classes: number of classes in the dataset
    :param kernel_regularizer: kernel regularization for the encoder
    :param projection_dim: embedding dimension
    :param encoder: class of the encoder
    :param encoder_per_path: whether to have individual encoders or shared weights
    :param image_size: size of the image to resize to
    :param patch_size: size of the patch
    :param pathway_classification: whether the classification is done for independent classifiers (even when it is not a classifier, it is usefull to have this metric during training)
    :param pathway_classification_allpaths: if pathways classification, whether each encoder should be trained in a supervised manner, or only the first
    :param ensemble_classification: whether a linear classifier from the concatenated output of all encoders should learn to classify
    :param classifier_pathways: if false, the classifier receives the full image and is a single encoder
    :param augmentation_kwargs: kwargs for get_data_augmentation
    :param encoder_kwargs: kwargs for the encoder
    :param pathways_kwargs: kwargs for SplitPathwaysVision
    :param patch_encoder: whether to use Patch encoding
    :param patch_encoder_after_split: if true, the patches are encoded after splitting, so that the linear projection is not shared
    :param label_to_dim: ignore
    :return: a tensorflow Model
    """


    if isinstance(kernel_regularizer, str) and kernel_regularizer.startswith("tf."):
        kernel_regularizer = eval(kernel_regularizer)

    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = get_data_augmentation(image_size, **augmentation_kwargs)(inputs)

    if len(augmented.shape) == 3:
        augmented = augmented[..., None]

    if divide_patches:
        # Create patches
        patches = Patches(patch_size, name=name + '_patch')(augmented)
        num_patches = (image_size // patch_size) ** 2
    else:
        if divide_rgb:
            patches = tf.reshape(augmented, (-1, augmented.shape[-3]*augmented.shape[-2], augmented.shape[-1]))
        else:
            patches = tf.keras.layers.Flatten(name=name + "_flatten")(augmented)[..., None]
        num_patches = patches.shape[-2]

    # Encode patches.
    if not patch_encoder_after_split:
        num_class_tokens = pathways_kwargs.get('n', 2) if pathways_kwargs.get('token_per_path', False) else (max(eval(pathways_kwargs.get('pathway_to_cls', '[0]'))) + 1)
        patches = PatchEncoder(num_patches, projection_dim, name=name + '_patchenc',
                               kernel_regularizer=kernel_regularizer,
                               num_class_tokens=num_class_tokens)(patches) if patch_encoder else patches

    # divide to different pathways
    if classifier and not classifier_pathways:
        pathways = [patches]
    else:
        # generate the input for each encoder with a corresponding class token
        pathways = SplitPathwaysVision(num_patches, class_token=patch_encoder and not patch_encoder_after_split,
                                       name=name + '_pathways', **pathways_kwargs)(patches)
        pathways = [tf.squeeze(path, axis=-2) for path in tf.split(pathways, pathways.shape[-2], axis=-2)]
        if patch_encoder and patch_encoder_after_split:
            assert encoder_per_path
            pathways = [PatchEncoder(p.shape[1], projection_dim, name=name + f'_patchenc{i}',
                                     kernel_regularizer=kernel_regularizer, num_class_tokens=1)(p)
                        for i,p in enumerate(pathways)]

    # encoders
    import utils.model.encoders
    Encoder = get_class(encoder, utils.model.encoders)
    out_reg = tf.keras.regularizers.L2(l2) if l2 else None
    enc_init = lambda i: Encoder(name=name+f'_enc{i if i is not None else ""}',
                                 kernel_regularizer=kernel_regularizer, out_regularizer=out_reg, **encoder_kwargs)
    encoders = [enc_init(i) for i in range(len(pathways))] if encoder_per_path else [enc_init(None)] * len(pathways)

    embedding = tf.keras.layers.Concatenate(name=name + '_embedding', axis=-1)([encoder(pathway)[..., None]
                                                                                for encoder, pathway in zip(encoders, pathways)])

    outputs = [embedding]


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
    """
    Compiles the model

    :param model: a tensorflow Model
    :param loss: the embedding loss class
    :param loss_kwargs: kwargs for the loss init
    :param optimizer_cls: class of the optimizer
    :param optimizer_kwargs: kwargs for the optimizer init
    :param classifier: whether the model is supervised or not
    :param pathway_classification: whether the classification is done for independent classifiers (even when it is not a classifier, it is usefull to have this metric during training)
    :param ensemble_classification: whether a linear classifier from the concatenated output of all encoders should learn to classify
    :param pathway_classification_allpaths: if pathways classification, whether each encoder should be trained in a supervised manner, or only the first
    :param metrics_kwargs: kwargs for the metrics
    :param kwargs: leftover kwargs that won't be used
    :return: None
    """
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
            # see utils\metrics\LossMonitors
            if model.name + '_embedding' not in metrics:
                metrics[model.name + '_embedding'] = []
            elif not isinstance(metrics[model.name + '_embedding'], list):
                metrics[model.name + '_embedding'] = [metrics['embedding']]
            for k, m in loss.monitor.monitors.items():
                metrics[model.name + '_embedding'].append(m)

    optimizer = get_optimizer(optimizer_cls=optimizer_cls, **optimizer_kwargs)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
