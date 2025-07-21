from utils.data import GeneratorDataset
from utils.model.callbacks import SaveOptimizerCallback, ErasePreviousCallback, SaveHistory, StopIfNaN
from utils.model.losses import NullLoss
from utils.model.optimizer import load_optimizer
from utils.modules import Modules
from utils.tf_utils import get_weights_fn
from utils.utils import printd
import os
import tensorflow as tf


def create_and_compile_model(model_name, dataset, model_kwargs, module: Modules, loss=NullLoss, loss_kwargs={},
                             optimizer_kwargs={}, metrics_kwargs={}, print_log=False, **kwargs):
    """
    a module-general function which inits a model and compiles it
    :param model_name: name of the model, which can be used to get the corresponding json file
    :param dataset: any utils/data class
    :param model_kwargs: kwargs for the model creation
    :param module: Module
    :param loss: embedding loss name or class
    :param loss_kwargs: embedding loss kwargs for init
    :param optimizer_kwargs: optimizer kwargs for init
    :param metrics_kwargs: metrics kwargs
    :param print_log: whether to print progress comments
    :param kwargs: leftover kwargs for both create_model and compile_model
    :return:
    """
    if print_log:
        printd("Creating model...", end='\t')
    m = module.create_model(name=model_name, input_shape=dataset.get_shape(),
                            label_to_dim=dataset.get_label_to_dim() if hasattr(dataset, "get_label_to_dim") else None,
                            **model_kwargs, **kwargs)
    if print_log:
        printd("Done!")

    loss = module.get_loss(loss)

    if print_log:
        printd("Compiling model...", end='\t')
    module.compile_model(m, dataset=dataset, loss=loss, loss_kwargs=loss_kwargs, optimizer_kwargs=optimizer_kwargs,
                         metrics_kwargs=metrics_kwargs, **kwargs)

    return m


def load_or_create_model(model_name, module: Modules, *args, load=True, optimizer_state=True,
                         skip_mismatch=False, pretrained_name=None, **kwargs):
    """
    a module-general function which load or creates a new model
    :param model_name: name of the model
    :param module: Module
    :param args: args for "create_and_compile_model"
    :param load: whether to load if the model exists. Defaults to true
    :param optimizer_state: whether to load the optimizer state. Defaults to true
    :param skip_mismatch: when loading weights, whether to ignore mismatches
    :param pretrained_name: a name of a different model to use as pre-trained weights
    :param kwargs: kwargs for "create_and_compile_model"
    :return: (a tensorflow Model, number of epochs it was already trained)
    """
    import re

    model = create_and_compile_model(model_name, *args, module=module, **kwargs)
    max_epoch = 0
    if load:    # try to load weights
        model_fn = None
        dir_path = os.path.join(module.get_models_path(), f"{model_name}/checkpoints")
        if os.path.exists(dir_path):
            for fn in os.listdir(dir_path):
                # find the latest weights
                match = re.match(r"model_weights_(\d+)\.index", fn)
                if match:
                    epoch = int(match.group(1))
                    if epoch > max_epoch:
                        max_epoch = epoch
                        model_fn = f"model_weights_{max_epoch}"
            if model_fn:    # if found weights, load them
                printd(f"loading checkpoint {model_fn}")
                if optimizer_state:
                    load_optimizer(model, module)
                model.load_weights(os.path.join(dir_path, model_fn),
                                   skip_mismatch=skip_mismatch, by_name=skip_mismatch)
        if not max_epoch:
            printd("didn't find previous checkpoint")

    if pretrained_name is not None and not max_epoch:
        # load weight
        printd(f"trying to load from {pretrained_name}")
        pretrained_model = load_model_from_json(pretrained_name, module)
        for i, l in enumerate(model.layers):
            if len(l.weights) == 0:
                continue
            pretrained_layer_name = l.name.replace(model.name, pretrained_model.name)
            loaded_w = False
            for layer in pretrained_model.layers:
                if layer.name == pretrained_layer_name:
                    try:
                        l.set_weights([w.numpy() for w in layer.weights])
                        printd(f"loaded layer {l.name}")
                    except Exception as err:
                        printd(f"couldn't load layer {l.name}, got exception {err}")
                    loaded_w = True
                    break
            if not loaded_w:
                printd(f"couldn't load layer {l.name}, name didn't exist")
    return model, max_epoch


def load_model_from_json(model_name, module: Modules, load=True, optimizer_state=True, skip_mismatch=False):
    dct = module.load_json(model_name, config=True)
    if dct is None:
        return None
    else:
        def call(dataset, data_kwargs={}, **kwargs):
            model, _ = load_or_create_model(model_name, module=module,
                                            dataset=module.get_class_from_data(dataset)(module=module,**data_kwargs),
                                            load=load, optimizer_state=optimizer_state, skip_mismatch=skip_mismatch,
                                            **kwargs)
            return model

        return call(**dct)


def train(model_name, module: Modules, data_kwargs={}, dataset="Cifar10", batch_size=128, num_epochs=150, **kwargs):
    """
    A module-general training function
    :param model_name: name of the model
    :param module: name of the module
    :param data_kwargs: kwargs for the data init
    :param dataset: name of the dataset
    :param batch_size: batch size
    :param num_epochs: number of epochs to train (if already trained K, then training for num_of_epochs-K)
    :param kwargs: from the configuration file
    :return:
    """

    # checking a gpu is available
    printd("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    printd(f"{tf.test.is_gpu_available()=}")
    printd(f"{tf.test.is_built_with_cuda()=}")

    # loading the dataset
    printd("Getting dataset...", end='\t')
    dataset = module.get_class_from_data(dataset)(module=module, **data_kwargs)
    printd("Done!")

    # loading\creating the model
    model, max_epoch = load_or_create_model(model_name, module, dataset=dataset, print_log=True, **kwargs)

    model.summary()

    if num_epochs > max_epoch:
        printd(f"Fitting the model (with {model.count_params()} parameters)!")

        fit_kwargs = dict(epochs=num_epochs,
                          initial_epoch=max_epoch,
                          callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=get_weights_fn(model, module),
                                                                        save_weights_only=True,
                                                                        save_best_only=False,
                                                                        verbose=1),
                                     SaveOptimizerCallback(module), ErasePreviousCallback(module),
                                     SaveHistory(module), StopIfNaN(module)]
                          )

        if issubclass(dataset.__class__, GeneratorDataset):
            val_dataset = dataset.get_val()
            # fitting the model
            history = model.fit(iter(dataset),
                                validation_data=iter(val_dataset),
                                batch_size=dataset.batch_size,
                                steps_per_epoch=1000,
                                validation_steps=250,
                                **fit_kwargs)
        else:
            # fitting the model
            history = model.fit(x=dataset.get_x_train(),
                                y=dataset.get_y_train(),
                                validation_split=dataset.get_val_split() if hasattr(dataset, 'get_val_split') else None,
                                validation_data=None if hasattr(dataset, 'get_val_split') else (
                                dataset.get_x_val(), dataset.get_y_val()),
                                batch_size=batch_size,
                                **fit_kwargs)

        printd("Done!")
    return model
