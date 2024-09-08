import tensorflow as tf

from utils.modules import Modules
from utils.tf_utils import history_fn_name


class SaveOptimizerCallback(tf.keras.callbacks.Callback):
    def __init__(self, module: Modules, *args, **kwargs):
        super(SaveOptimizerCallback, self).__init__(*args, **kwargs)
        self.module = module

    def on_epoch_end(self, epoch, logs=None):
        import pickle
        import os
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
        with open(os.path.join(self.module.get_models_path(), self.model.name,
                               "checkpoints", 'optimizer.pkl'), 'wb') as f:
            pickle.dump(weight_values, f)


class ErasePreviousCallback(tf.keras.callbacks.Callback):
    def __init__(self, module: Modules, *args, **kwargs):
        super(ErasePreviousCallback, self).__init__(*args, **kwargs)
        self.module = module

    def on_epoch_end(self, epoch, logs=None):
        import os
        fns = [f"model_weights_{epoch}.data-00000-of-00001", f"model_weights_{epoch}.index"]
        fns = [os.path.join(self.module.get_models_path(), f"{self.model.name}/checkpoints/" + fn)
               for fn in fns]
        for fn in fns:
            if os.path.exists(fn):
                os.remove(fn)


class SaveHistory(tf.keras.callbacks.Callback):
    def __init__(self, module:Modules):
        super().__init__()
        self.history = None
        self.module = module
        self.epoch = []

    def on_train_begin(self, logs=None):
        self.epoch = []
        prev_history = self.module.load_json(history_fn_name(self.model.name))
        self.history = {} if prev_history is None else prev_history

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.module.save_json(self.module.history_fn_name(self.model.name), self.history)


class SaveHistoryWithMetrics(SaveHistory):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for loss_name, loss in self.model.loss.items():
            # Access the loss values from the model
            if hasattr(loss, 'get_metrics'):
                for metric_name, metric in loss.get_metrics():
                    logs[metric_name] = metric.numpy()

        super().on_epoch_end(epoch, logs=logs)


class HistoryWithMetrics(tf.keras.callbacks.History):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for loss_name, loss in self.model.loss.items():
            # Access the loss values from the model
            if hasattr(loss, 'get_metrics'):
                for metric_name, metric in loss.get_metrics():
                    logs[metric_name] = metric.numpy()

        super().on_epoch_end(epoch, logs=logs)
