import neuronal.model.model as neur_model
import neuronal.model.losses as neur_losses
from auditory.utils.data import Labels
from utils.modules import Modules


def create_model(input_shape, name='auditory_model', encoder='TimeAgnosticMLP',
                 labels=(Labels.BIRD, ), module=Modules.AUDITORY, **kwargs):
    labels = [eval(label) if isinstance(label, str) else label for label in labels]
    return neur_model.create_model(input_shape, name=name, encoder=encoder, labels=labels, module=module, **kwargs)


def compile_model(model, dataset, loss=neur_losses.LPL, labels=(Labels.BIRD, ), **kwargs):
    labels = [eval(label) if isinstance(label, str) else label for label in labels]
    return neur_model.compile_model(model, dataset, loss=loss, labels=labels, **kwargs)
