from typing import Optional

from utils.data import Data
from utils.evaluation.ensemble import EnsembleVotingMethods
from utils.evaluation.evaluation import classify_head_eval_ensemble
from utils.model.layers import SplitPathways
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import printd
from vision.utils.data import Cifar10
import numpy as np
from tqdm import tqdm as counter

EVAL_IND_LIN = 'pathways_mean_linear'
EVAL_IND_LIN_NAME = 'Single encoder'
EVAL_ENS_LIN = 'logistic'
EVAL_ENS_LIN_NAME = 'Ensemble'


def get_masked_ds(model, dataset=Cifar10()) -> Data:
    """
    given a model, returns a Data object with the patchified and masked images
    :param model: tensorflow Model or str
    :param dataset: an utils/data class
    :return: a Data object
    """
    if isinstance(model, str):
        model = load_model_from_json(model, Modules.VISION)
    aug_layer = model.get_layer("data_augmentation")
    patch_layer = model.get_layer(model.name + '_patch')
    pathway_indices = model.get_layer(model.name + '_pathways').indices.numpy()
    setup_func = lambda x: np.transpose(patch_layer(aug_layer(x)).numpy()[:, pathway_indices - model.get_layer(model.name + '_pathways').shift], [0, 1, 3, 2]).reshape(
        x.shape[0], -1, pathway_indices.shape[-1])
    ds = Data(setup_func(dataset.get_x_train()), dataset.get_y_train(),
              setup_func(dataset.get_x_test()), dataset.get_y_test(),
              x_val=setup_func(dataset.get_x_val()) if dataset.get_x_val() is not None else None,
              y_val=dataset.get_y_val(), normalize=True)
    return ds


def evaluate(model, module: Modules=Modules.VISION, linear=True, ensemble=True,
             override_linear=False, save_results=False, override=False, inp=True,
             dataset:Optional[Data]=Cifar10(), **kwargs):
    """
    Evaluate the model (classification accuracy)
    :param model: tensorflow Model or str
    :param module: Module
    :param linear: whether to evaluate the ensemble (embedding concatenation)
    :param ensemble: whether to evaluate individual encoders and ensembling methods (not the basic embedding concatenation)
    :param override_linear: override previous results for ensemble embedding concatenation
    :param save_results: whether to save the results
    :param override: overrider all previous results
    :param inp: evaluate the classification accuracy for the input baselines
    :param dataset: the dataset to use. If not given, assumes it's the one from the model's configuration file
    :param kwargs: further evaluation kwargs
    :return: {evaluation name: (train score, val score, test score)}
    """

    if isinstance(model, str):
        # load the model
        model_kwargs = module.load_json(model, config=True)
        assert model_kwargs is not None
        model = load_model_from_json(model, module)
        if not model_kwargs['model_kwargs']['pathways_kwargs'].get('fixed', False):
            pathways:SplitPathways = model.get_layer(model.name + "_pathways")
            pathways.fixed = True
            pathways.get_indices()
        if dataset is None:
            dataset_cls = module.get_class_from_data(model_kwargs.get('dataset', 'Cifar10'))
            dataset = dataset_cls(module=module, **model_kwargs.get('data_kwargs', {}), split=True)

    printd("getting embedding...")
    printd("train...", end='\t')
    x_train_embd = model.predict(dataset.get_x_train())[0]
    printd("done!")

    printd("test...", end='\t')
    x_test_embd = model.predict(dataset.get_x_test())[0]
    printd("done!")

    printd("validation...", end='\t')
    x_val_embd = model.predict(dataset.get_x_val())[0]
    printd("done!")

    embd_dataset = Data(x_train_embd, dataset.get_y_train(), x_test_embd, dataset.get_y_test(),
                        x_val=x_val_embd, y_val=dataset.get_y_val(), normalize=True)

    from utils.evaluation.evaluation import classify_head_eval

    results = {} if override and not override_linear else module.load_evaluation_json(model.name)

    if results is None:
        results = {}

    save_res = lambda *inputs: module.save_evaluation_json(model.name, results) if save_results else None

    # run the embedding concatenation ensemble decoding
    if linear:
        if 'logistic' not in results or override_linear:
            printd("running logistic")
            results[EVAL_ENS_LIN] = classify_head_eval(embd_dataset, linear=True, svm=False, categorical=True, **kwargs)
            save_res()

    if ensemble:
        # run individual encoders and ensembleing methods (that are not presented in the paper)
        printd("running ensemble")
        results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False, categorical=True,
                                                   voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb], **kwargs))
        save_res()
        if inp:
            # evaluate the classification accuracy for the input baselines
            printd("running inp")
            if not any([k.startswith("image_pathway") for k in results.keys()]):
                masked_ds = get_masked_ds(model, dataset=dataset)
                results.update(classify_head_eval_ensemble(masked_ds, base_name='image_', svm=False, categorical=True,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb]), **kwargs)
                save_res()
    return results
