from model.layers import *
from model.losses import *
from utils.data import *
from utils.tf_utils import get_model_fn, save_model, load_model_from_json
from utils.utils import *
import utils.data
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


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


def create_model(name='model', only_classtoken=False, koleo_lambda=0, classifier=False, l2=False,
                 input_shape=(32, 32, 3), num_classes=10, transformer_layers=3,
                 projection_dim=64, out_block_kwargs={}, block_kwargs={}, pathways_kwargs={},
                 image_size=72, patch_size=8):
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = get_data_augmentation(image_size)(inputs)

    # Create patches.
    patches = Patches(patch_size, name=name + '_patch')(augmented)
    num_patches = (image_size // patch_size) ** 2

    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim, name=name + '_patchenc',
                                   num_class_tokens=pathways_kwargs.get('n', 2) if pathways_kwargs.get('token_per_path', False) else 1)(patches)

    # divide to different pathways
    if classifier:
      pathways = [encoded_patches]
    else:
      pathways = SplitPathways(num_patches, name=name + '_pathways', **pathways_kwargs)(encoded_patches)
      pathways = [tf.squeeze(path, axis=-2) for path in tf.split(pathways, pathways.shape[-2], axis=-2)]

    # process each pathway
    blocks = [ViTBlock(name=name + f'_block{l}', **block_kwargs) for l in range(transformer_layers)]
    for l in range(transformer_layers):
      for i in range(len(pathways)):
        pathways[i] = blocks[l](pathways[i])

    # save only the class token
    if only_classtoken:
      for i in range(len(pathways)):
        pathways[i] = pathways[i][:, 0]

    # out block
    out_block = ViTOutBlock(name=name + '_outblock', **out_block_kwargs,
                            activity_regularizer=KoLeoRegularizer(koleo_lambda) if koleo_lambda else (tf.keras.regularizers.L2(l2) if l2 else None))
    embedding = tf.keras.layers.Concatenate(name=name + '_embedding', axis=-1)([out_block(enc)[..., None] for enc in pathways])

    # classification head
    logits = layers.Dense(num_classes, activation=None, name=name + '_logits')(embedding[..., 0] if classifier else tf.stop_gradient(embedding[..., 0]))

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=[embedding, logits], name=name)
    return model


def compile_model(model, loss=ContrastiveSoftmaxLoss, loss_kwargs={}, optimizer_cls=tf.optimizers.Nadam,
                  optimizer_kwargs={}, classifier=False):
    optimizer = optimizer_cls(**optimizer_kwargs)
    serialize(optimizer.__class__, 'Custom')

    model.compile(
        optimizer=optimizer,
        loss={
            model.name + '_embedding': loss(**loss_kwargs) if not classifier else lambda *args: 0.,
            model.name + '_logits': keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
        metrics={model.name + '_logits': keras.metrics.SparseCategoricalAccuracy(name="accuracy")}
    )


def train(model_name, model_kwargs, loss=ContrastiveSoftmaxLoss, loss_kwargs={},
          optimizer_kwargs={},
          classifier=False, dataset=Cifar10, batch_size=128, num_epochs=150):
    printd("Getting dataset...", end='\t')
    dataset = get_class(dataset, utils.data)()
    printd("Done!")

    printd("Checking if model already trained...", end='\t')
    model = load_model_from_json(model_name)
    if model is not None:
        printd("Loaded model!")
    else:
        printd("Model never trained before")
        printd("Creating model...", end='\t')
        model = create_model(model_name, input_shape=dataset.get_shape(), **model_kwargs)
        printd("Done!")

        loss = get_class(loss, model.losses)

        printd("Compiling model...", end='\t')
        compile_model(model, loss=loss, loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs, classifier=classifier)
        printd("Done!")

    # TODO: regression callback?

    if num_epochs:
        printd("Fitting the model!")
        history = model.fit(
            x=dataset.get_x_train(),
            y=dataset.get_y_train(),
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=dataset.get_val_split(),
            callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=get_model_fn(model),
                                                          save_weights_only=False,
                                                          verbose=1)]
        )

        printd("saving the model!")
        save_model(model)
        printd("Done!")
    return model
