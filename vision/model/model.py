from model.callbacks import SaveOptimizerCallback, ErasePreviousCallback, SaveHistory
from model.layers import *
from model.losses import *
from utils.data import *
from utils.io_utils import load_json, save_json
from utils.tf_utils import get_model_fn, get_weights_fn
from utils.utils import *
import utils.data
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

PATHWAY_TO_CLS = None


def get_data_augmentation(image_size):
    data_augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(image_size, image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    return data_augmentation


def create_model(name='model', koleo_lambda=0, classifier=False, l2=False,
                 input_shape=(32, 32, 3), num_classes=10, kernel_regularizer=None,
                 projection_dim=64, encoder='ViTEncoder', encoder_per_path=False,
                 encoder_kwargs={}, pathways_kwargs={}, image_size=72, patch_size=8,
                 pathway_classification=True, ensemble_classification=False, classifier_pathways=True):
    if isinstance(kernel_regularizer, str) and kernel_regularizer.startswith("tf."):
        kernel_regularizer = eval(kernel_regularizer)


    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = get_data_augmentation(image_size)(inputs)

    if len(augmented.get_shape()) == 3:
        augmented = augmented[..., None]

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
        pathways = SplitPathways(num_patches, name=name + '_pathways', **pathways_kwargs)(encoded_patches)
        pathways = [tf.squeeze(path, axis=-2) for path in tf.split(pathways, pathways.shape[-2], axis=-2)]

    import model.encoders
    Encoder = get_class(encoder, model.encoders)
    out_reg = KoLeoRegularizer(koleo_lambda) if koleo_lambda else (tf.keras.regularizers.L2(l2) if l2 else None)
    enc_init = lambda i: Encoder(name=name+f'_enc{i if i is not None else ""}',
                                 kernel_regularizer=kernel_regularizer, out_regularizer=out_reg, **encoder_kwargs)
    encoders = [enc_init(i) for i in range(len(pathways))] if encoder_per_path else [enc_init(None)] * len(pathways)

    embedding = tf.keras.layers.Concatenate(name=name + '_embedding', axis=-1)([encoder(pathway)[..., None]
                                                                                for encoder, pathway in zip(encoders, pathways)])

    outputs = [embedding]

    # classification heads, with stop_grad unless classifier=True
    if pathway_classification:
        outputs.append(layers.Dense(num_classes, activation=None, kernel_regularizer=kernel_regularizer, name=name + '_logits')(
            (embedding if classifier else tf.stop_gradient(embedding))[..., 0]))
    if ensemble_classification:
        outputs.append(layers.Dense(num_classes, activation=None, kernel_regularizer=kernel_regularizer, name=name + '_ensemble_logits')(
            tf.reshape(embedding if classifier else tf.stop_gradient(embedding), (-1, np.multiply.reduce(embedding.shape[1:])))))

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_model(model, loss=ContrastiveSoftmaxLoss, loss_kwargs={}, optimizer_cls=tf.optimizers.legacy.Nadam if tf.__version__ == '2.12.0' else tf.optimizers.Nadam,
                  optimizer_kwargs={}, classifier=False, pathway_classification=True,
                  ensemble_classification=False, **kwargs):
    if kwargs:
        print(f"WARNING: compile_model got spare kwargs that won't be used: {kwargs}")

    optimizer = optimizer_cls(**optimizer_kwargs)
    serialize(optimizer.__class__, 'Custom')

    losses = {}
    metrics = {}
    if classifier:
        losses[model.name + '_embedding'] = NullLoss()
    else:
        losses[model.name + '_embedding'] = loss(**loss_kwargs)

    if pathway_classification:
        losses[model.name + '_logits'] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics[model.name + '_logits'] = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    if ensemble_classification:
        losses[model.name + '_ensemble_logits'] = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics[model.name + '_ensemble_logits'] = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)


def train(model_name, model_kwargs, loss=ContrastiveSoftmaxLoss, data_kwargs={}, loss_kwargs={},
          optimizer_kwargs={}, dataset=Cifar10, batch_size=128, num_epochs=150, **kwargs):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    printd("Getting dataset...", end='\t')
    dataset = get_class(dataset, utils.data)(**data_kwargs)
    printd("Done!")

    model, max_epoch = load_or_create_model(model_name, dataset.get_shape(), model_kwargs, loss=loss,
                                            loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs,
                                            print_log=True, **kwargs)

    # TODO: regression callback?

    if num_epochs > max_epoch:
        printd("Fitting the model!")
        history = model.fit(
            x=dataset.get_x_train(),
            y=dataset.get_y_train(),
            batch_size=batch_size,
            epochs=num_epochs,
            initial_epoch=max_epoch,
            validation_split=dataset.get_val_split(),
            callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=get_weights_fn(model),
                                                          save_weights_only=True,
                                                          save_best_only=False,
                                                          verbose=1),
                       SaveOptimizerCallback(), ErasePreviousCallback(), SaveHistory()]
        )
        printd("Done!")
    return model


def create_and_compile_model(model_name, input_shape, model_kwargs, loss=ContrastiveSoftmaxLoss, loss_kwargs={},
                             optimizer_kwargs={}, print_log=False, **kwargs):
    if print_log:
        printd("Creating model...", end='\t')
    m = create_model(model_name, input_shape=input_shape, **model_kwargs, **kwargs)
    if print_log:
        printd("Done!")

    import model.losses
    loss = get_class(loss, model.losses)

    if print_log:
        printd("Compiling model...", end='\t')
    compile_model(m, loss=loss, loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs, **kwargs)

    return m


def load_or_create_model(model_name, *args, load=True, optimizer_state=True, skip_mismatch=False, **kwargs):
    import os
    import re

    model = create_and_compile_model(model_name, *args, **kwargs)
    max_epoch = 0
    if load:
        model_fn = None
        if os.path.exists(f"models/{model_name}/checkpoints"):
            for fn in os.listdir(f'models/{model_name}/checkpoints'):
                match = re.match(r"model_weights_(\d+)\.index", fn)
                if match:
                    epoch = int(match.group(1))
                    if epoch > max_epoch:
                        max_epoch = epoch
                        model_fn = f"model_weights_{max_epoch}"
            if model_fn:
                print(f"loading checkpoint {model_fn}")
                if optimizer_state:
                    load_optimizer(model)
                model.load_weights(os.path.join("models", model_name, "checkpoints", model_fn),
                                   skip_mismatch=skip_mismatch, by_name=skip_mismatch)
        if not max_epoch:
            print("didn't find previous checkpoint")
    return model, max_epoch


def load_model_from_json(model_name, load=True, optimizer_state=True, skip_mismatch=False):
    dct = load_json(model_name)
    if dct is None:
        return None
    else:

        def call(model_kwargs, loss=ContrastiveSoftmaxLoss, loss_kwargs={}, optimizer_kwargs={},
                 dataset=Cifar10, data_kwargs={}, **kwargs):
            dataset = get_class(dataset, utils.data)(**data_kwargs)
            model, _ = load_or_create_model(model_name, dataset.get_shape(), model_kwargs, loss=loss,
                                            loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs,
                                            load=load, optimizer_state=optimizer_state, skip_mismatch=skip_mismatch, **kwargs)
            return model

        return call(**dct)


def load_optimizer(model):
    import os
    import pickle
    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
    with open(os.path.join("models", model.name, "checkpoints", 'optimizer.pkl'), 'rb') as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)
