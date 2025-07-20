from tensorflow.keras.losses import Loss


class NullLoss(Loss):
    def __init__(self, *args, name='NullLoss', **kwargs):
        super(NullLoss, self).__init__(*args, name=name, **kwargs)

    def call(self, y_true, y_pred):
        return 0.
