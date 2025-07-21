import tensorflow as tf


class LossMonitor(tf.keras.metrics.MeanMetricWrapper):
    """
    Used to have multiple metrics for the same output .
    See for example in vision/utils/compile_model
    """
    def __init__(self, name='loss_monitor'):
        super().__init__(fn=self.get_current_value, name=name)
        self.value = self.add_weight(name='value', initializer='zeros')

    def get_current_value(self, *args, **kwargs):
        return self.value

    def update_monitor(self, value):
        self.value.assign(value)


class LossMonitors:
    def __init__(self, *names, name='loss_monitors'):
        self.monitors = {n: LossMonitor(name=(name + "_" if name else '') + n) for n in names}

    def update_monitor(self, name, value):
        self.monitors[name].update_monitor(value)