from auditory.utils.data import Labels
import neuronal.evaluate as neur_evaluate
from utils.modules import Modules


def evaluate(model, dataset=None, module: Modules=Modules.AUDITORY, labels=[Labels.BIRD], **kwargs):
    labels = [eval(label) if isinstance(label, str) else label for label in labels]
    return neur_evaluate.evaluate(model=model, dataset=dataset, module=module, labels=labels, alltime=True, **kwargs)
