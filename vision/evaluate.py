from typing import Optional

from utils.data import Data
from utils.evaluation.ensemble import EnsembleVotingMethods
from utils.evaluation.evaluation import classify_head_eval_ensemble
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import printd
from vision.utils.data import Cifar10
import numpy as np


def get_masked_ds(model, dataset=Cifar10()):
    if isinstance(model, str):
        model = load_model_from_json(model, Modules.VISION)
    aug_layer = model.get_layer("data_augmentation")
    patch_layer = model.get_layer(model.name + '_patch')
    pathway_indices = model.get_layer(model.name + '_pathways').indices.numpy()
    setup_func = lambda x: np.transpose(patch_layer(aug_layer(x)).numpy()[:, pathway_indices - model.get_layer(model.name + '_pathways').shift], [0, 1, 3, 2]).reshape(
        x.shape[0], -1, pathway_indices.shape[-1])
    ds = Data(setup_func(dataset.get_x_train()), dataset.get_y_train(),
              setup_func(dataset.get_x_test()), dataset.get_y_test())
    return ds


def evaluate(model, module: Modules=Modules.VISION, knn=False, linear=True, ensemble=True, ensemble_knn=False,
             save_results=False, override=False, inp=True, dataset:Optional[Data]=Cifar10(), ks=[1] + list(range(5, 50, 5)), **kwargs):

    if isinstance(model, str):
        model_kwargs = module.load_json(model, config=True)
        assert model_kwargs is not None
        model = load_model_from_json(model, module)
        if dataset is None:
            dataset = module.get_class_from_data(model_kwargs.get('dataset', 'Cifar10'))(**model_kwargs.get('data_kwargs', {}))

    x_train_embd = model.predict(dataset.get_x_train())[0]
    x_test_embd = model.predict(dataset.get_x_test())[0]
    embd_dataset = Data(x_train_embd, dataset.get_y_train(), x_test_embd, dataset.get_y_test())

    from utils.evaluation.evaluation import classify_head_eval

    results = module.load_evaluation_json(model.name) if not override else {}

    if results is None:
        results = {}

    save_res = lambda *inputs: module.save_evaluation_json(model.name, results) if save_results else None

    if knn:
        for k in ks:
            if f'k={k}' not in results:
                printd(f"k={k}:", end='\t')
                results[f"k={k}"] = classify_head_eval(embd_dataset, linear=False, k=k, **kwargs)
                save_res()

    if linear:
        if 'logistic' not in results:
            results['logistic'] = classify_head_eval(embd_dataset, linear=True, svm=False, **kwargs)
            save_res()

    if ensemble:
        results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False,
                                                   voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb], **kwargs))
        save_res()
        if inp:
            if not any([k.startswith("image_pathway") for k in results.keys()]):
                masked_ds = get_masked_ds(model, dataset=dataset)
                results.update(classify_head_eval_ensemble(masked_ds, base_name='image_', svm=False,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb]), **kwargs)
                save_res()

    if ensemble_knn:
        results.update(classify_head_eval_ensemble(embd_dataset, linear=False, svm=False, k=15,
                                                   voting_methods=EnsembleVotingMethods), **kwargs)
        save_res()
    return results
