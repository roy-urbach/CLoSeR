import tensorflow as tf
from tf.keras import layers

from utils.model.layers import SplitPathways
from utils.utils import get_class
from vision.utils.tf_utils import set_seed, serialize


def get_neuronal_data_augmentation(inputs, **kwargs):
    # TODO: implement
    return inputs


class SplitPathwaysNeuronal(SplitPathways):
    """
    Call
    :param inputs: (B, T, N)
    :return: (B, T, K, d*N)
    """

    def __init__(self, num_units, n=2, d=0.5, intersection=True, fixed=False, seed=0, **kwargs):
        super(SplitPathwaysNeuronal, self).__init__(num_channels=num_units, n=n, d=d, intersection=intersection,
                                                    fixed=fixed, seed=seed, class_token=False, **kwargs)
        self.num_units = num_units


def create_neuronal_model(name='neuronal_model', frames=9, bins_per_frame=1, units=100,
                          classifier=False, l2=False, kernel_regularizer=None,
                          projection_dim=64, encoder='BasicRNN', encoder_per_path=False,
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
        pathways = SplitPathwaysNeuronal(units, name=name + '_pathways', **pathways_kwargs)(augmented)
        pathways = [tf.squeeze(path, axis=-2) for path in tf.split(pathways, pathways.shape[-2], axis=-2)]

    import utils.model.encoders
    Encoder = get_class(encoder, utils.model.encoders)
    out_reg = tf.keras.regularizers.L2(l2) if l2 else None
    enc_init = lambda i: Encoder(name=name+f'_enc{i if i is not None else ""}',
                                 kernel_regularizer=kernel_regularizer, out_regularizer=out_reg, **encoder_kwargs)
    encoders = [enc_init(i) for i in range(len(pathways))] if encoder_per_path else [enc_init(None)] * len(pathways)

    embedding = tf.keras.layers.Concatenate(name=name + '_embedding', axis=-1)([encoder(pathway)[..., None]
                                                                                for encoder, pathway in zip(encoders, pathways)])
    # (B, T, C, P)

    outputs = [embedding]

    last_step_embedding = embedding[:, -1]

    # # classification heads, with stop_grad unless classifier=True
    # if pathway_classification:
    #     # Only for the first pathway
    #     embedding_for_classification = last_step_embedding if classifier else tf.stop_gradient(last_step_embedding)
    #     pathways_to_classify = list(range(len(pathways))) if pathway_classification_allpaths else [0]
    #     for path in pathways_to_classify:
    #         cur_embd = embedding_for_classification[..., path]
    #         cur_name = name + '_logits' + (str(path) if pathway_classification_allpaths else '')
    #         pathway_logits = layers.Dense(num_classes, activation=None,
    #                                       kernel_regularizer=kernel_regularizer, name=cur_name)(cur_embd)
    #         outputs.append(pathway_logits)
    # if ensemble_classification:
    #     outputs.append(layers.Dense(num_classes, activation=None, kernel_regularizer=kernel_regularizer, name=name + '_ensemble_logits')(
    #         tf.reshape(embedding if classifier else tf.stop_gradient(embedding), (-1, np.multiply.reduce(embedding.shape[1:])))))

    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


