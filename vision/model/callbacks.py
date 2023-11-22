import tensorflow as tf
from utils.io_utils import save_json


class SaveOptimizerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, *args, logs=None, **kwargs):
        import pickle
        import os
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
        with open(os.path.join("models", self.model.name, "checkpoints", 'optimizer.pkl'), 'wb') as f:
            pickle.dump(weight_values, f)


class ErasePreviousCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, **kwargs):
        import os
        fns = [f"model_weights_{epoch-1}.data-00000-of-00001", f"model_weights_{epoch-1}.index"]
        fns = [f"models/{self.model.name}/checkpoints/" + fn for fn in fns]
        for fn in fns:
            if os.path.exists(fn):
                os.remove(fn)


class SaveHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        save_json("history", self.history, base_path=f"models/{self.mode.name}/checkpoints/")