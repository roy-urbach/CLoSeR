from neuronal.utils.data import Labels, CATEGORICAL
from utils.data import Data
from utils.evaluation.ensemble import EnsembleVotingMethods
from utils.evaluation.evaluation import classify_head_eval_ensemble
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import printd, streval, run_on_dict, flatten_but_batch
import numpy as np


def get_masked_ds(model, dataset, union=False, bins_per_frame=1, last_frame=True, normalize=False,
                  flatten=True, module=Modules.NEURONAL):
    if isinstance(model, str):
        model = load_model_from_json(model, module)

    dct_out = False

    if isinstance(dataset.get_x_train(), dict) and 'pathways' not in [l.name for l in model.layers]:
        if union:
            setup_func = lambda x: np.concatenate([arr[..., -bins_per_frame if last_frame else 0:] for arr in x.values()], axis=-2)
        else:
            dct_out = True
            setup_func = lambda x: run_on_dict(x, lambda a: a[..., -bins_per_frame if last_frame else 0:])  # still dict

    else:
        aug_layer = model.get_layer("data_augmentation")
        pathways = model.get_layer('pathways')
        pathway_indices = pathways.indices.numpy()
        if union:
            union_inds = np.unique(pathway_indices) - model.get_layer("pathways").shift
            setup_func = lambda x: aug_layer(x).numpy()[:, union_inds, ..., -bins_per_frame if last_frame else 0:]   # (B, N, T)
        else:
            setup_func = lambda x: np.transpose(aug_layer(x).numpy()[:, pathway_indices - pathways.shift, ..., -bins_per_frame if last_frame else 0:], [0, 1, 3, 2]).reshape(
                x.shape[0], -1, bins_per_frame if last_frame else x.shape[-1], pathway_indices.shape[-1])       # (B, N, T, P)

    x_train = setup_func(dataset.get_x_train())
    x_test = setup_func(dataset.get_x_test())
    x_val = setup_func(dataset.get_x_val())

    if flatten:
        if union:
            x_train = flatten_but_batch(x_train)
            x_val = flatten_but_batch(x_val)
            x_test = flatten_but_batch(x_test)
        elif dct_out:
            x_train = run_on_dict(x_train, flatten_but_batch)
            x_val = run_on_dict(x_val, flatten_but_batch)
            x_test = run_on_dict(x_test, flatten_but_batch)
        else:
            x_train = x_train.reshape((len(x_train), -1, x_train.shape[-1]))
            x_val = x_val.reshape((len(x_val), -1, x_val.shape[-1]))
            x_test = x_test.reshape((len(x_test), -1, x_test.shape[-1]))

    ds = Data(x_train, dataset.get_y_train(),
              x_test, dataset.get_y_test(),
              x_val=x_val, y_val=dataset.get_y_val(), normalize=normalize, flatten_y=False)
    return ds


def evaluate(model, dataset=None, module: Modules=Modules.NEURONAL, labels=[Labels.STIMULUS],
             linear=True, ensemble=True, save_results=False, override=False, override_linear=False,
             inp=True, model_kwargs=None, only_input=False, **kwargs):
    """
    Evaluate neuronal models
    :param model: name or tf Model object
    :param dataset: utils\data class object or None. If model is not the model name, must be given too.
    :param module: the module. Default: NEURONAL
    :param labels: a list of labels to evaluate on
    :param linear: whether to train a head using the concatenation of the encoders' output
    :param ensemble: whether to train a head for every encoder and also ensembling methods
    :param save_results: whether to save the results
    :param override: whether to override previous results
    :param override_linear: whether to override previous results for the embedding concatenation ensemble
    :param inp: whether to evaluate on the input too
    :param model_kwargs: the configuration dictionary. overrides the model name if given
    :param only_input: whether to evaluate only the input
    :param kwargs: head evaluation kwargs
    :return: {metric: (train, val, test)} for all metrics if validation data exists, otherwise (train, test)
    """

    if isinstance(model, str):
        model_kwargs = module.load_json(model, config=True) if model_kwargs is None else model_kwargs
        assert model_kwargs is not None
        printd("loading model...", end='\t')
        model = load_model_from_json(model, module, load=not only_input)
        printd("done")
        if dataset is None:
            printd("loading dataset...", end='\t')
            dataset = module.get_class_from_data(model_kwargs['dataset'])(module=module, **model_kwargs.get('data_kwargs', {}))
            printd("done")
    assert dataset is not None

    bins_per_frame = model.bins_per_frame if hasattr(model, "bins_per_frame") else (dataset.bins_per_frame if hasattr(dataset, 'bins_per_frame') else 1)
    encoder_removed_bins = model.layers[0].output_shape[-1] not in (model.get_layer("embedding").output_shape[1],
                                                                    model.get_layer("embedding").output_shape[1] // 2)
    if not only_input:

        def transform_embedding(embedding, last_frame=True):
            if last_frame:
                if encoder_removed_bins:
                    last_step_embedding = embedding[:, -1]
                else:
                    last_step_embedding = embedding[:, -bins_per_frame:]    # (B, bins_per_frame, DIM, P)
                    last_step_embedding = last_step_embedding.reshape(last_step_embedding.shape[0],
                                                                      last_step_embedding.shape[-2] * bins_per_frame,
                                                                      last_step_embedding.shape[-1])  # (B, DIMS*bins_per_frame, P)
                return last_step_embedding
            else:
                return embedding.reshape(embedding.shape[0], embedding.shape[-2] * embedding.shape[-3], embedding.shape[-1])

        printd("getting predictions...", end='\t')
        x_train_embd = model.predict(dataset.get_x_train())[0]
        x_test_embd = model.predict(dataset.get_x_test())[0]
        x_train_embd_flattened = transform_embedding(x_train_embd)

        x_test_embd_flattened = transform_embedding(x_test_embd)

        x_val = dataset.get_x_val()
        if x_val is not None:
            x_val_embd = model.predict(x_val)[0]
            x_val_embd_flattened = transform_embedding(x_val_embd)
        else:
            x_val_embd = None
            x_val_embd_flattened = None

    labels = [module.get_label(label) for label in streval(labels)]

    def get_inp_ds(last_frame=False, label=None, union=False, flatten=True, cache={}):
        nonflattened_name = f"lastframe{last_frame}_un{union}"
        flattened_name = nonflattened_name + f"_flatten{flatten}"
        name = flattened_name if flatten else nonflattened_name
        if name in cache:
            ds = cache[name]
        else:
            if flatten and nonflattened_name in cache:
                ds = cache[nonflattened_name]
            else:
                ds = get_masked_ds(model, dataset=dataset, bins_per_frame=bins_per_frame,
                                   union=union, module=module,
                                   last_frame=last_frame, normalize=True, flatten=False)
                cache[nonflattened_name] = ds
            cur_x_train, cur_x_test, cur_x_val = ds.get_x_train(), ds.get_x_test(), ds.get_x_val()

            if flatten:
                if union:
                    cur_x_train = cur_x_train.reshape(len(cur_x_train), -1)
                    cur_x_val = cur_x_val.reshape(len(cur_x_val), -1)
                    cur_x_test = cur_x_test.reshape(len(cur_x_test), -1)
                elif isinstance(cur_x_train, dict):
                    cur_x_train = run_on_dict(cur_x_train, flatten_but_batch)
                    cur_x_val = run_on_dict(cur_x_val, flatten_but_batch)
                    cur_x_test = run_on_dict(cur_x_test, flatten_but_batch)
                else:
                    cur_x_train = cur_x_train.reshape((len(cur_x_train), -1, cur_x_train.shape[-1]))
                    cur_x_val = cur_x_val.reshape((len(cur_x_val), -1, cur_x_val.shape[-1]))
                    cur_x_test = cur_x_test.reshape((len(cur_x_test), -1, cur_x_test.shape[-1]))

            ds = Data(cur_x_train, dataset.get_y_train(labels=module.get_label(label).value.name) if label is not None else None,
                      cur_x_test, dataset.get_y_test(labels=module.get_label(label).value.name) if label is not None else None,
                      x_val=cur_x_val, y_val=dataset.get_y_val(labels=module.get_label(label).value.name) if label is not None else None,
                      normalize=False, flatten_y=False)

            cache[name] = ds

        return ds
    printd("done")

    results = module.load_evaluation_json(model.name) if not override else {}

    if results is None:
        results = {}

    save_res = lambda *inputs: module.save_evaluation_json(model.name, results) if save_results else None

    # iterate over all labels to evaluate
    for label in labels:
        y_train = dataset.get_y_train(labels=label.value.name)
        y_test = dataset.get_y_test(labels=label.value.name)
        y_val = dataset.get_y_val(labels=label.value.name) if x_val_embd_flattened is not None else None

        if not only_input:
            embd_dataset = Data(x_train_embd_flattened, y_train,
                                x_test_embd_flattened, y_test,
                                x_val=x_val_embd_flattened, y_val=y_val,
                                normalize=True)
        else:
            embd_dataset = None

        from utils.evaluation.evaluation import classify_head_eval

        if linear:
            # embedding concatenation ensemble evaluation
            if not only_input and (f'{label.value.name}_linear' not in results or override_linear):
                printd("linear")
                results[f'{label.value.name}_linear'] = classify_head_eval(embd_dataset,
                                                                           categorical=label.value.kind == CATEGORICAL,
                                                                           linear=True, svm=False, **kwargs)
                save_res()

            # input concatenation ensemble evaluation
            if inp and (f'{label.value.name}_input_linear' not in results or override_linear):
                printd("input linear")
                results[f'{label.value.name}_input_linear'] = classify_head_eval(get_inp_ds(last_frame=True, label=label.value.name, union=True),
                                                                                 categorical=label.value.kind == CATEGORICAL,
                                                                                 linear=True, svm=False, **kwargs)
                save_res()

        if ensemble:
            # individual encoders and ensembling methods
            if not only_input and (not any(['linear' in k and 'ensemble' in k and label.value.name in k for k in results]) or override_linear):
                printd("ensemble linear")
                results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False,
                                                           base_name=f"{label.value.name}_",
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean], **kwargs))
                save_res()

            # individual inputs and ensembling methods on the input
            if inp and (not any([k.startswith(f"{label.value.name}_input_pathway") for k in results.keys()]) or override_linear):
                printd("ensemble input linear")
                results.update(classify_head_eval_ensemble(get_inp_ds(last_frame=True, label=label.value.name, union=False),
                                                           base_name=f"{label.value.name}_input_", svm=False,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                save_res()
    return results
