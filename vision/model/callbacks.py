import tensorflow as tf
from utils.io_utils import save_json, load_json
from utils.tf_utils import history_fn_name


class SaveOptimizerCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        import pickle
        import os
        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
        with open(os.path.join("models", self.model.name, "checkpoints", 'optimizer.pkl'), 'wb') as f:
            pickle.dump(weight_values, f)


class ErasePreviousCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        import os
        fns = [f"model_weights_{epoch}.data-00000-of-00001", f"model_weights_{epoch}.index"]
        fns = [f"models/{self.model.name}/checkpoints/" + fn for fn in fns]
        for fn in fns:
            if os.path.exists(fn):
                os.remove(fn)


class SaveHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.history = None

    def on_train_begin(self, logs=None):
        self.epoch = []
        prev_history = load_json(history_fn_name(self.model.name), base_path='')
        self.history = {} if prev_history is None else prev_history

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        save_json(history_fn_name(self.model.name), self.history, base_path="")