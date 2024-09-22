from typing import Optional

from neuronal.utils.data import Labels, CATEGORICAL
from utils.data import Data
from utils.evaluation.ensemble import EnsembleVotingMethods
from utils.evaluation.evaluation import classify_head_eval_ensemble
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import printd
import numpy as np


def get_masked_ds(model, dataset, union=False, bins_per_frame=1):
    if isinstance(model, str):
        model = load_model_from_json(model, Modules.NEURONAL)
    aug_layer = model.get_layer("data_augmentation")
    pathways = model.get_layer('pathways')
    pathway_indices = pathways.indices.numpy()
    if union:
        union = np.unique(pathway_indices) - model.get_layer("pathways").shift
        setup_func = lambda x: aug_layer(x).numpy()[:, union, ..., -bins_per_frame:]
    else:
        setup_func = lambda x: np.transpose(aug_layer(x).numpy()[:, pathway_indices - pathways.shift, ..., -bins_per_frame:], [0, 1, 3, 2]).reshape(
            x.shape[0], -1, pathway_indices.shape[-1])

    ds = Data(setup_func(dataset.get_x_train()), dataset.get_y_train(),
              setup_func(dataset.get_x_test()), dataset.get_y_test(),
              x_val=setup_func(dataset.get_x_val()), y_val=dataset.get_y_val())
    return ds


def evaluate(model, dataset="SessionDataGenerator", module: Modules=Modules.NEURONAL, labels=[Labels.STIMULUS, Labels.FRAME],
             knn=False, linear=True, ensemble=True, save_results=False, override=False, override_linear=False, **kwargs):

    if isinstance(model, str):
        model_kwargs = module.load_json(model, config=True)
        assert model_kwargs is not None
        model = load_model_from_json(model, module)
        if dataset is None:
            dataset = module.get_class_from_data(model_kwargs['dataset'])(**model_kwargs.get('data_kwargs', {}))

    bins_per_frame = dataset.bins_per_frame
    def transform_embedding(embedding):
        encoder_removed_bins = model.get_layer("pathways").output_shape[-1] != model.get_layer("embedding").output_shape[1]
        if encoder_removed_bins:
            last_step_embedding = embedding[:, -1]
        else:
            last_step_embedding = embedding[:, -bins_per_frame:]    # (B, bins_per_frame, DIM, P)
            last_step_embedding = last_step_embedding.reshape(last_step_embedding.shape[0],
                                                              last_step_embedding.shape[-2] * bins_per_frame,
                                                              last_step_embedding.shape[-1])  # (B, DIMS*bins_per_frame, P)
        return last_step_embedding

    x_train_embd = transform_embedding(model.predict(dataset.get_x_train())[0])
    x_test_embd = transform_embedding(model.predict(dataset.get_x_test())[0])
    x_val = dataset.get_x_val()
    if x_val is not None:
        x_val_embd = transform_embedding(model.predict(x_val)[0])
    else:
        x_val_embd = None

    y_train = dataset.get_y_train(labels)
    y_test = dataset.get_y_test(labels)
    y_val = dataset.get_y_val(labels)


    results = module.load_evaluation_json(model.name) if not override else {}

    if results is None:
        results = {}

    save_res = lambda *inputs: module.save_evaluation_json(model.name, results) if save_results else None

    for label in labels:
        print(f"evaluating label {label.value.name}")
        basic_dataset = Data(dataset.get_x_train(), y_train[label.value.name],
                             dataset.get_x_test(), y_test[label.value.name],
                             x_val=dataset.get_x_val(), y_val=y_val[label.value.name] if y_val is not None else None)

        embd_dataset = Data(x_train_embd, y_train[label.value.name],
                            x_test_embd, y_test[label.value.name],
                            x_val=x_val_embd, y_val=y_val[label.value.name] if y_val is not None else None)

        from utils.evaluation.evaluation import classify_head_eval

        if knn:
            for k in [1] + list(range(5, 21, 5)):
                cur_name = f"{label.value.name}_k={k}"
                if cur_name not in results:
                    printd(cur_name, ":", end='\t')
                    results[cur_name] = classify_head_eval(embd_dataset,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           linear=False, k=k, **kwargs)
                    save_res()
                cur_name = f"{label.value.name}_input_k={k}"
                if cur_name not in results:
                    printd(cur_name, ":", end='\t')
                    results[cur_name] = classify_head_eval(get_masked_ds(model, dataset=basic_dataset, union=True),
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           linear=False, k=k, **kwargs)
                    save_res()

        if linear:
            if f'{label.value.name}_linear' not in results or override_linear:
                results[f'{label.value.name}_linear'] = classify_head_eval(embd_dataset,
                                                                           categorical=label.value.kind == CATEGORICAL,
                                                                           linear=True, svm=False, **kwargs)
                save_res()

            if f'{label.value.name}_input_linear' not in results or override_linear:
                results[f'{label.value.name}_input_linear'] = classify_head_eval(get_masked_ds(model, dataset=basic_dataset, union=True),
                                                                                 categorical=label.value.kind == CATEGORICAL,
                                                                                 linear=True, svm=False, **kwargs)
                save_res()

        if ensemble:
            results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False,
                                                       base_name=f"{label.value.name}_",
                                                       categorical=label.value.kind == CATEGORICAL,
                                                       voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean], **kwargs))
            save_res()

            if not any([k.startswith(f"{label.value.name}_input_pathway") for k in results.keys()]) or override_linear:
                masked_ds = get_masked_ds(model, dataset=basic_dataset)
                results.update(classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_input_", svm=False,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                save_res()

        if ensemble and knn:
            for k in [1] + list(range(5, 21, 5)):
                results.update(classify_head_eval_ensemble(embd_dataset, linear=False, svm=False, k=k,
                                                           base_name=f"{label.value.name}_k={k}_",
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                save_res()

                masked_ds = get_masked_ds(model, dataset=basic_dataset)
                results.update(classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_input_k={k}_",
                                                                svm=False, k=k, linear=False,
                                                                categorical=label.value.kind == CATEGORICAL,
                                                                voting_methods=[
                                                                    EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                                **kwargs))
                save_res()
    return results
