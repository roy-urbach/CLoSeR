import tensorflow as tf

from utils.modules import Modules


def get_optimizer(optimizer_cls=tf.optimizers.legacy.Nadam if tf.__version__ == '2.12.0' else tf.optimizers.Nadam,
                  scheduler_kwargs={}, weight_decay=None, **kwargs):
    """
    creates an optimizer
    :param optimizer_cls:
    :param scheduler_kwargs:
    :param weight_decay:
    :param kwargs:
    :return:
    """
    if scheduler_kwargs:
        assert "scheduler" in scheduler_kwargs
        scheduler_name = scheduler_kwargs['scheduler']
        try:
            from tensorflow.keras.optimizers import schedules
            scheduler_cls = getattr(schedules, scheduler_name)
        except Exception as err:
            print(f"Tried to getattr tf.keras.optimizers.schedules.{scheduler_name}, get error:", err)
            raise err
        kwargs['learning_rate'] = scheduler_cls(**{k: v for k, v in scheduler_kwargs.items() if k != 'scheduler'})

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

    if weight_decay:
        optimizer = WeightDecayOptimizer(optimizer, weight_decay=weight_decay)
    return optimizer


class WeightDecayOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, weight_decay):
        self.optimizer = optimizer
        super().__init__(name="wd_" + str(optimizer))
        for k, v in self.optimizer.__dict__.items():
            if k not in ("apply_gradients", "get_config"):
                setattr(self, k, v)
        self.weight_decay = weight_decay

    def apply_gradients(self, grads_and_vars, name=None):
        self.optimizer.apply_gradients(grads_and_vars, name=name)

        for _, var in grads_and_vars:
            lr = self.optimizer._decayed_lr(var.dtype)  # Get the current learning rate
            var.assign(var * (1 - self.weight_decay * lr))

    def get_config(self):
        config = self.optimizer.get_config()
        config.update({'weight_decay': self.weight_decay})
        return config



def load_optimizer(model, module: Modules):
    import os
    import pickle
    grad_vars = model.trainable_weights
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
    with open(os.path.join(module.get_models_path(), model.name, "checkpoints", 'optimizer.pkl'), 'rb') as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)
