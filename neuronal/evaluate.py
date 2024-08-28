from typing import Optional

from neuronal.utils.data import Labels, CATEGORICAL
from utils.data import Data
from utils.evaluation.ensemble import EnsembleVotingMethods
from utils.evaluation.evaluation import classify_head_eval_ensemble
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import printd
import numpy as np


def get_masked_ds(model, dataset):
    if isinstance(model, str):
        model = load_model_from_json(model, Modules.NEURONAL)
    aug_layer = model.get_layer("data_augmentation")
    pathway_indices = model.get_layer('pathways').indices.numpy()
    setup_func = lambda x: np.transpose(aug_layer(x).numpy()[:, pathway_indices - model.get_layer('pathways').shift], [0, 1, 3, 2]).reshape(
        x.shape[0], -1, pathway_indices.shape[-1])
    ds = Data(setup_func(dataset.get_x_train()), dataset.get_y_train(),
              setup_func(dataset.get_x_test()), dataset.get_y_test())
    return ds


def evaluate(model, dataset="SessionDataGenerator", module: Modules=Modules.NEURONAL, labels=[Labels.STIMULUS, Labels.FRAME],
             knn=False, linear=True, ensemble=True, ensemble_knn=False, save_results=False, override=False, **kwargs):

    if isinstance(model, str):
        model_kwargs = module.load_json(model, config=True)
        assert model_kwargs is not None
        model = load_model_from_json(model, module)
        if dataset is None:
            dataset = module.get_class_from_data(model_kwargs['dataset'])(**model_kwargs.get('data_kwargs', {}))

    test_dataset = dataset.get_test()

    bins_per_frame = dataset.bins_per_frame
    def transform_embedding(embedding):
        last_step_embedding = embedding[:, -bins_per_frame:]    # (B, bins_per_frame, DIM, P)
        last_step_embedding = last_step_embedding.reshape(last_step_embedding.shape[0],
                                                          last_step_embedding.shape[-2] * bins_per_frame,
                                                          last_step_embedding.shape[-1])  # (B, DIMS*bins_per_frame, P)
        return last_step_embedding

    x_train_embd = transform_embedding(model.predict(dataset.get_x())[0])
    x_test_embd = transform_embedding(model.predict(test_dataset.get_x())[0])

    y_train = dataset.get_y(labels)
    y_test = dataset.get_y(labels)

    results = module.load_evaluation_json(model.name) if not override else {}

    if results is None:
        results = {}

    save_res = lambda *inputs: module.save_evaluation_json(model.name, results) if save_results else None

    for label in labels:

        embd_dataset = Data(x_train_embd, y_train[label.value.name], x_test_embd, y_test[label.value.name])

        from utils.evaluation.evaluation import classify_head_eval

        if knn:
            for k in [1] + list(range(5, 50, 5)):
                if f'k={k}' not in results:
                    printd(f"{label.value.name}_k={k}:", end='\t')
                    results[f"{label.value.name}_k={k}"] = classify_head_eval(embd_dataset,
                                                                              categorical=label.value.kind == CATEGORICAL,
                                                                              linear=False, k=k, **kwargs)
                    save_res()

        if linear:
            if f'{label.value.name}_linear' not in results:
                results[f'{label.value.name}_linear'] = classify_head_eval(embd_dataset,
                                                                           categorical=label.value.kind == CATEGORICAL,
                                                                           linear=True, svm=False, **kwargs)
                save_res()

        if ensemble:
            results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False,
                                                       base_name=f"{label.value.name}_",
                                                       categorical=label.value.kind == CATEGORICAL,
                                                       voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb], **kwargs))
            save_res()

            if not any([k.startswith(f"{label.value.name}_input_pathway") for k in results.keys()]):
                masked_ds = get_masked_ds(model, dataset=dataset)
                results.update(classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_input_", svm=False,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb]), **kwargs)
                save_res()

        if ensemble_knn:
            results.update(classify_head_eval_ensemble(embd_dataset, linear=False, svm=False, k=15,
                                                       voting_methods=EnsembleVotingMethods), **kwargs)
            save_res()
    return results
