from utils.data import gen_to_tf_dataset
from utils.model.callbacks import SaveOptimizerCallback, SaveHistory, ErasePreviousCallback, AddLossMetrics
from utils.model.losses import NullLoss
from utils.modules import Modules
from utils.tf_utils import get_weights_fn, serialize
from utils.utils import printd
import os
import tensorflow as tf


def get_optimizer(optimizer_cls=tf.optimizers.legacy.Nadam if tf.__version__ == '2.12.0' else tf.optimizers.Nadam,
                  **kwargs):
    if "optimizer" in kwargs:
        optimizer_cls_name = kwargs.pop("optimizer")
        try:
            optimizer_cls = eval(optimizer_cls_name)
        except Exception as err:
            print(f"Tried to eval {optimizer_cls_name}, get error:", err)
            try:
                optimizer_cls = tf.keras.optimizers.getattr(optimizer_cls_name)
            except Exception as err:
                print(f"Tried to getattr tf.keras.optimizers.{optimizer_cls_name}, get error:", err)
                raise err
    print(f"using optimizer {optimizer_cls}")
    optimizer = optimizer_cls(**{k: eval(v) if isinstance(v, str) and v.startswith("tf.") else v
                                 for k, v in kwargs.items()})
    serialize(optimizer.__class__, 'Custom')
    return optimizer


def create_and_compile_model(model_name, dataset, model_kwargs, module: Modules, loss=NullLoss, loss_kwargs={},
                             optimizer_kwargs={}, metrics_kwargs={}, print_log=False, **kwargs):
    if print_log:
        printd("Creating model...", end='\t')
    m = module.create_model(name=model_name, input_shape=dataset.get_shape(), **model_kwargs, **kwargs)
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
    import re

    model = create_and_compile_model(model_name, *args, module=module, **kwargs)
    max_epoch = 0
    if load:
        model_fn = None
        dir_path = os.path.join(module.get_models_path(), f"{model_name}/checkpoints")
        if os.path.exists(dir_path):
            for fn in os.listdir(dir_path):
                match = re.match(r"model_weights_(\d+)\.index", fn)
                if match:
                    epoch = int(match.group(1))
                    if epoch > max_epoch:
                        max_epoch = epoch
                        model_fn = f"model_weights_{max_epoch}"
            if model_fn:
                print(f"loading checkpoint {model_fn}")
                if optimizer_state:
                    load_optimizer(model, module)
                model.load_weights(os.path.join(dir_path, model_fn),
                                   skip_mismatch=skip_mismatch, by_name=skip_mismatch)
        if not max_epoch:
            print("didn't find previous checkpoint")

    if pretrained_name is not None and not max_epoch:
        print(f"trying to load from {pretrained_name}")
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
                        print(f"loaded layer {l.name}")
                    except Exception as err:
                        print(f"could load layer {l.name}, got exception {err}")
                    loaded_w = True
                    break
            if not loaded_w:
                print(f"couldn't load layer {l.name}, name didn't exist")
    return model, max_epoch


def load_model_from_json(model_name, module: Modules, load=True, optimizer_state=True, skip_mismatch=False):
    dct = module.load_json(model_name, config=True)
    if dct is None:
        return None
    else:
        def call(dataset, data_kwargs={}, **kwargs):
            model, _ = load_or_create_model(model_name, module, dataset=module.get_class_from_data(dataset)(**data_kwargs),
                                            load=load, optimizer_state=optimizer_state, skip_mismatch=skip_mismatch,
                                            **kwargs)
            return model

        return call(**dct)


def load_optimizer(model, module: Modules):
    import os
    import pickle
    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
    with open(os.path.join(module.get_models_path(), model.name, "checkpoints", 'optimizer.pkl'), 'rb') as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)


def train(model_name, module: Modules, data_kwargs={}, dataset="Cifar10", batch_size=128, num_epochs=150, **kwargs):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    printd("Getting dataset...", end='\t')
    dataset = module.get_class_from_data(dataset)(**data_kwargs)
    printd("Done!")

    model, max_epoch = load_or_create_model(model_name, module, dataset=dataset, print_log=True, **kwargs)

    # TODO: regression callback?

    if num_epochs > max_epoch:
        printd(f"Fitting the model (with {model.count_params()} parameters)!")
        fit_kwargs = dict(epochs=num_epochs,
                          initial_epoch=max_epoch,
                          callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=get_weights_fn(model, module),
                                                                        save_weights_only=True,
                                                                        save_best_only=False,
                                                                        verbose=1),
                                     SaveOptimizerCallback(module), ErasePreviousCallback(module),
                                     AddLossMetrics(), SaveHistory(module)]
                          )
        if dataset.is_generator():
            dataset = gen_to_tf_dataset(dataset, batch_size=batch_size, buffer_size=batch_size)
            val_dataset = dataset.get_validation()
            val_dataset.random = False
            dataset.random = False
            history = model.fit(x=dataset.get_x(),
                                y=dataset.get_y(),
                                validation_data=(val_dataset.get_x(), val_dataset.get_y()),
                                # steps_per_epoch=int(dataset.__len__() / batch_size),
                                # validation_steps=int(val_dataset.__len__() / batch_size),
                                **fit_kwargs)
        else:
            history = model.fit(x=dataset.get_x_train(),
                                y=dataset.get_y_train(),
                                validation_split=dataset.get_val_split(),
                                batch_size=batch_size,
                                **fit_kwargs)
        printd("Done!")
    return model
