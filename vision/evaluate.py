from typing import Optional

from utils.data import Data
from utils.evaluation.ensemble import EnsembleVotingMethods
from utils.evaluation.evaluation import classify_head_eval_ensemble
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import printd
from vision.utils.data import Cifar10
import numpy as np
from tqdm import tqdm as counter

EVAL_IND_LIN = 'pathways_mean_linear'
EVAL_ENS_LIN = 'logistic'


def get_masked_ds(model, dataset=Cifar10()):
    if isinstance(model, str):
        model = load_model_from_json(model, Modules.VISION)
    aug_layer = model.get_layer("data_augmentation")
    patch_layer = model.get_layer(model.name + '_patch')
    pathway_indices = model.get_layer(model.name + '_pathways').indices.numpy()
    setup_func = lambda x: np.transpose(patch_layer(aug_layer(x)).numpy()[:, pathway_indices - model.get_layer(model.name + '_pathways').shift], [0, 1, 3, 2]).reshape(
        x.shape[0], -1, pathway_indices.shape[-1])
    ds = Data(setup_func(dataset.get_x_train()), dataset.get_y_train(),
              setup_func(dataset.get_x_test()), dataset.get_y_test(),
              x_val=setup_func(dataset.get_x_val()), y_val=dataset.get_y_val(), normalize=True)
    return ds


def evaluate(model, module: Modules=Modules.VISION, knn=False, linear=True, ensemble=True, ensemble_knn=False,
             override_linear=False, save_results=False, override=False, inp=True,
             dataset:Optional[Data]=Cifar10(), ks=[1] + list(range(5, 50, 5)), **kwargs):

    if isinstance(model, str):
        model_kwargs = module.load_json(model, config=True)
        assert model_kwargs is not None
        model = load_model_from_json(model, module)
        if dataset is None:
            dataset = module.get_class_from_data(model_kwargs.get('dataset', 'Cifar10'))(module=module, **model_kwargs.get('data_kwargs', {}), split=True)

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

    if knn:
        for k in ks:
            if f'k={k}' not in results:
                printd(f"k={k}:", end='\t')
                results[f"k={k}"] = classify_head_eval(embd_dataset, linear=False, k=k, categorical=True, **kwargs)
                save_res()

    if linear:
        if 'logistic' not in results or override_linear:
            printd("running logistic")
            results[EVAL_ENS_LIN] = classify_head_eval(embd_dataset, linear=True, svm=False, categorical=True, **kwargs)
            save_res()

    if ensemble:
        printd("running ensemble")
        results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False, categorical=True,
                                                   voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb], **kwargs))
        save_res()
        if inp:
            printd("running inp")
            if not any([k.startswith("image_pathway") for k in results.keys()]):
                masked_ds = get_masked_ds(model, dataset=dataset)
                results.update(classify_head_eval_ensemble(masked_ds, base_name='image_', svm=False, categorical=True,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb]), **kwargs)
                save_res()

    if ensemble_knn:
        printd("running ensemble knn")
        results.update(classify_head_eval_ensemble(embd_dataset, linear=False, svm=False, k=15,
                                                   categorical=True,
                                                   voting_methods=EnsembleVotingMethods), **kwargs)
        save_res()
    return results


def evaluate_k(model, module, k, repeats, **kwargs):
    model_kwargs = module.load_json(model, config=True)
    assert model_kwargs is not None
    model = load_model_from_json(model, module)
    dataset = module.get_class_from_data(model_kwargs.get('dataset', 'Cifar10'))(module=module,
                                                                                 **model_kwargs.get('data_kwargs', {}),
                                                                                 split=True)

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

    P = x_train_embd.shape[-1]

    embd_dataset = Data(x_train_embd, dataset.get_y_train(), x_test_embd, dataset.get_y_test(),
                        x_val=x_val_embd, y_val=dataset.get_y_val(), normalize=True)
    save_res = lambda *inputs: module.save_evaluation_json_k(model.name, results, k=k)

    from utils.evaluation.evaluation import classify_head_eval
    results = {}
    for rep in counter(range(repeats)):
        chosen_ps = np.random.permutation(P)[:k].astype(np.int32)
        results.setdefault('k', []).append([int(num) for num in chosen_ps])
        cur_ds = Data(embd_dataset.get_x_train()[..., chosen_ps], dataset.get_y_train(),
                      embd_dataset.get_x_test()[..., chosen_ps], dataset.get_y_test(),
                      x_val=embd_dataset.get_x_val()[..., chosen_ps], y_val=dataset.get_y_val(), normalize=False)
        results.setdefault('res', []).append(classify_head_eval(cur_ds, linear=True, svm=False, categorical=True, **kwargs))
        save_res()

    return results
