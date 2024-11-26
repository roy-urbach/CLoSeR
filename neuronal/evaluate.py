from typing import Optional

from neuronal.utils.data import Labels, CATEGORICAL
from utils.data import Data
from utils.evaluation.ensemble import EnsembleVotingMethods
from utils.evaluation.evaluation import classify_head_eval_ensemble, classify_head_eval
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import printd, streval
import numpy as np


def get_masked_ds(model, dataset, union=False, bins_per_frame=1, last_frame=True, pcs=None, normalize=False, flatten=True):
    if isinstance(model, str):
        model = load_model_from_json(model, Modules.NEURONAL)
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

    if pcs is not None:
        from sklearn.decomposition import PCA
        pca = PCA(pcs)
        if union:
            pca.fit(np.concatenate([x_train[..., t] for t in range(x_train.shape[-1])], axis=0))
            x_train = np.stack(
                [pca.transform(x_train[..., t]) for t in range(x_train.shape[-1])], axis=-1)
            x_test = np.stack(
                [pca.transform(x_test[..., t]) for t in range(x_test.shape[-1])], axis=-1)
            x_val = np.stack(
                [pca.transform(x_val[..., t]) for t in range(x_val.shape[-1])], axis=-1)
        else:
            pca.fit(np.concatenate([x_train[...,t, i] for i in range(x_train.shape[-1]) for t in range(x_val.shape[-2])], axis=0))
            x_train = np.stack([np.stack([pca.transform(x_train[..., t, i])
                                          for t in range(x_train.shape[-2])], axis=-1)
                                for i in range(x_train.shape[-1])], axis=-1)
            x_test = np.stack([np.stack([pca.transform(x_test[..., t, i])
                                          for t in range(x_test.shape[-2])], axis=-1)
                                for i in range(x_test.shape[-1])], axis=-1)

            x_val = np.stack([np.stack([pca.transform(x_val[..., t, i])
                                          for t in range(x_val.shape[-2])], axis=-1)
                                for i in range(x_val.shape[-1])], axis=-1)

    if flatten:
        if union:
            x_train = x_train.reshape(len(x_train), -1)
            x_val = x_val.reshape(len(x_val), -1)
            x_test = x_test.reshape(len(x_test), -1)
        else:
            x_train = x_train.reshape((len(x_train), -1, x_train.shape[-1]))
            x_val = x_val.reshape((len(x_val), -1, x_val.shape[-1]))
            x_test = x_test.reshape((len(x_test), -1, x_test.shape[-1]))

    ds = Data(x_train, dataset.get_y_train(),
              x_test, dataset.get_y_test(),
              x_val=x_val, y_val=dataset.get_y_val(), normalize=normalize, flatten_y=False)
    return ds


def evaluate_predict(dct, masked_ds, masked_ds_union, embd_alltime_noflat_ds, encoder_removed_bins, bins_per_frame=1,
                     run_inp=True, base=''):
    print("evaluating predict...")
    func_y = lambda inp, union: inp[..., -bins_per_frame, :].reshape([len(inp), -1] + [inp.shape[-1]]*(not union))

    def get_ds(inp=False, union=False, alltime=False):

        y_ds = masked_ds_union if union else masked_ds
        x_ds = (masked_ds_union if union else masked_ds) if inp else embd_alltime_noflat_ds

        if x_ds is None:
            return None

        def transform_data(data):
            if inp:
                return data[..., (0 if alltime else -2*bins_per_frame):-bins_per_frame, :].reshape([len(data), -1] + [data.shape[-1]]*(not union))
            else:
                return data[..., (0 if alltime else -bins_per_frame):, :, :].reshape([len(data), -1] + [data.shape[-1]]*(not union))

        dataset = Data(
            transform_data(x_ds.get_x_train()), func_y(y_ds.get_x_train(), union=union),
            transform_data(x_ds.get_x_test()), func_y(y_ds.get_x_test(), union=union),
            x_val=transform_data(x_ds.get_x_val()), y_val=func_y(y_ds.get_x_val(), union=union),
            normalize=False
                       )

        return dataset

    for inp in (False, True):
        for alltime in (False, True):
            if not inp and embd_alltime_noflat_ds is None: continue
            if alltime and not inp and encoder_removed_bins: continue
            if inp and not run_inp: continue
            print(f"evaluating {inp=}, {alltime=}")
            ds = get_ds(inp=inp, alltime=alltime, union=False)
            if ds is None: continue
            dct.update(classify_head_eval_ensemble(ds,
                                                   base_name=f"predict_{'alltime_'*alltime}{'input_'*inp}",
                                                   linear=True, categorical=False, voting_methods=[None], svm=False, individual_ys=True))
            print(f"evaluating {inp=}, {alltime=}, linear")
            dct[f"predict_{(base + '_') if base else ''}{'alltime_'*alltime}{'input_'*inp}linear"] = classify_head_eval(get_ds(inp=inp, alltime=alltime, union=True),
                                                                                       categorical=False,
                                                                                       linear=True, svm=False)

    return dct


def evaluate(model, dataset=None, module: Modules=Modules.NEURONAL, labels=[Labels.STIMULUS],
             knn=False, linear=True, ensemble=True, save_results=False, override=False, override_linear=False, override_predict=False, inp=True,
             ks=[1] + list(range(5, 21, 5)), predict=False, only_input=False, pcs=[32, 64], **kwargs):

    if isinstance(model, str):
        model_kwargs = module.load_json(model, config=True)
        assert model_kwargs is not None
        printd("loading model...", end='\t')
        model = load_model_from_json(model, module, load=not only_input)
        printd("done")
        if dataset is None:
            printd("loading dataset...", end='\t')
            dataset = module.get_class_from_data(model_kwargs['dataset'])(**model_kwargs.get('data_kwargs', {}))
            printd("done")
    else:
        model_kwargs = module.load_json(model.name, config=True)

    bins_per_frame = dataset.bins_per_frame if hasattr(dataset, "bins_per_frame") else 1
    encoder_removed_bins = model.get_layer("pathways").output_shape[-1] not in (model.get_layer("embedding").output_shape[1],
                                                                                model.get_layer("embedding").output_shape[1] // 2)
    if not only_input:
        with_pred = "predictions" in [l.name for l in model.layers]

        def transform_embedding(embedding, last_frame=True):
            normalize = lambda arr: arr/np.linalg.norm(arr, keepdims=True, axis=-2) if model_kwargs['loss_kwargs'].pop("cosine", False) else arr

            if last_frame:
                if encoder_removed_bins:
                    last_step_embedding = normalize(embedding[:, -1])
                else:
                    last_step_embedding = normalize(embedding[:, -bins_per_frame:])    # (B, bins_per_frame, DIM, P)
                    last_step_embedding = last_step_embedding.reshape(last_step_embedding.shape[0],
                                                                      last_step_embedding.shape[-2] * bins_per_frame,
                                                                      last_step_embedding.shape[-1])  # (B, DIMS*bins_per_frame, P)
                return last_step_embedding
            else:
                return normalize(embedding).reshape(embedding.shape[0], embedding.shape[-2] * embedding.shape[-3], embedding.shape[-1])

        printd("getting predictions...", end='\t')
        x_train_embd = model.predict(dataset.get_x_train())[0]
        x_test_embd = model.predict(dataset.get_x_test())[0]
        x_train_embd_flattened = transform_embedding(x_train_embd)
        x_train_embd_flattened_alltime = transform_embedding(x_train_embd, last_frame=False)

        x_test_embd_flattened = transform_embedding(x_test_embd)
        x_test_embd_flattened_alltime = transform_embedding(x_test_embd, last_frame=False)

        x_val = dataset.get_x_val()
        if x_val is not None:
            x_val_embd = model.predict(x_val)[0]
            x_val_embd_flattened = transform_embedding(x_val_embd)
            x_val_embd_flattened_alltime = transform_embedding(x_val_embd, last_frame=False)
        else:
            x_val_embd = None
            x_val_embd_flattened = None
            x_val_embd_flattened_alltime = None

    labels = [Labels.get(label) for label in streval(labels)]

    def get_inp_ds(last_frame=False, pc=None, label=None, union=False, flatten=True, cache={}):
        nonflattened_name = f"lastframe{last_frame}_pc{pc}_un{union}"
        flattened_name = nonflattened_name + f"_flatten{flatten}"
        name = flattened_name if flatten else nonflattened_name
        if name in cache:
            ds = cache[name]
        else:
            if flatten and nonflattened_name in cache:
                ds = cache[nonflattened_name]
            else:
                ds = get_masked_ds(model, dataset=dataset, bins_per_frame=bins_per_frame,
                                   pcs=pc,
                                   last_frame=last_frame, normalize=True, flatten=False)
                cache[nonflattened_name] = ds
            cur_x_train, cur_x_test, cur_x_val = ds.get_x_train(), ds.get_x_test(), ds.get_x_val()

            if flatten:
                if union:
                    cur_x_train = cur_x_train.reshape(len(cur_x_train), -1)
                    cur_x_val = cur_x_val.reshape(len(cur_x_val), -1)
                    cur_x_test = cur_x_test.reshape(len(cur_x_test), -1)
                else:
                    cur_x_train = cur_x_train.reshape((len(cur_x_train), -1, cur_x_train.shape[-1]))
                    cur_x_val = cur_x_val.reshape((len(cur_x_val), -1, cur_x_val.shape[-1]))
                    cur_x_test = cur_x_test.reshape((len(cur_x_test), -1, cur_x_test.shape[-1]))

            ds = Data(cur_x_train, dataset.get_y_train(label) if label is not None else None,
                      cur_x_test, dataset.get_y_test(label) if label is not None else None,
                      x_val=cur_x_val, y_val=dataset.get_y_val(label) if label is not None else None, normalize=False, flatten_y=False)

            cache[name] = ds

        return ds
    printd("done")

    results = module.load_evaluation_json(model.name) if not override else {}

    if results is None:
        results = {}

    save_res = lambda *inputs: module.save_evaluation_json(model.name, results) if save_results else None

    for label in labels:
        y_train = dataset.get_y_train(label.value.name)
        y_test = dataset.get_y_test(label.value.name)
        y_val = dataset.get_y_val(label.value.name) if x_val_embd_flattened is not None else None

        print(y_train)

        printd(f"evaluating label {label.value.name}")
        if not only_input:
            embd_dataset = Data(x_train_embd_flattened, y_train,
                                x_test_embd_flattened, y_test,
                                x_val=x_val_embd_flattened, y_val=y_val,
                                normalize=True)
            embd_alltime_dataset = Data(x_train_embd_flattened_alltime, y_train,
                                        x_test_embd_flattened_alltime, y_test,
                                        x_val=x_val_embd_flattened_alltime, y_val=y_val,
                                        normalize=True)

            if with_pred:
                remove_pred = lambda embd: embd[..., :embd.shape[-2]//2, :]
                embd_dataset_nopred = Data(remove_pred(embd_dataset.get_x_train()), embd_dataset.get_y_train(),
                                           remove_pred(embd_dataset.get_x_test()), embd_dataset.get_y_test(),
                                           x_val=remove_pred(embd_dataset.get_x_val()), y_val=embd_dataset.get_y_val(),
                                           normalize=False
                                           )
                embd_alltime_dataset_nopred = Data(remove_pred(embd_alltime_dataset.get_x_train()), embd_alltime_dataset.get_y_train(),
                                           remove_pred(embd_alltime_dataset.get_x_test()), embd_alltime_dataset.get_y_test(),
                                           x_val=remove_pred(embd_alltime_dataset.get_x_val()), y_val=embd_alltime_dataset.get_y_val(),
                                           normalize=False
                                           )

        from utils.evaluation.evaluation import classify_head_eval

        if linear:
            if not only_input and (f'{label.value.name}_linear' not in results or override_linear):
                printd("linear")
                results[f'{label.value.name}_linear'] = classify_head_eval(embd_dataset,
                                                                           categorical=label.value.kind == CATEGORICAL,
                                                                           linear=True, svm=False, **kwargs)
                save_res()

            if not only_input and (with_pred and f'{label.value.name}_nopred_linear' not in results or override_linear):
                printd("nopred linear")
                results[f'{label.value.name}_nopred_linear'] = classify_head_eval(embd_dataset_nopred,
                                                                                  categorical=label.value.kind == CATEGORICAL,
                                                                                  linear=True, svm=False, **kwargs)
                save_res()

            if not only_input and dataset.frames_per_sample > 1:
                if f'{label.value.name}_alltime_linear' not in results or override_linear:
                    printd("alltime linear")
                    results[f'{label.value.name}_alltime_linear'] = classify_head_eval(embd_alltime_dataset,
                                                                                       categorical=label.value.kind == CATEGORICAL,
                                                                                       linear=True, svm=False, **kwargs)
                    save_res()

            if not only_input and (with_pred and f'{label.value.name}_nopred_alltime_linear' not in results or override_linear):
                printd("nopred alltime linear")
                results[f'{label.value.name}_nopred_alltime_linear'] = classify_head_eval(embd_alltime_dataset_nopred,
                                                                                  categorical=label.value.kind == CATEGORICAL,
                                                                                  linear=True, svm=False, **kwargs)
                save_res()

            if inp and (f'{label.value.name}_input_linear' not in results or override_linear):
                printd("input linear")
                results[f'{label.value.name}_input_linear'] = classify_head_eval(get_inp_ds(last_frame=True, pc=None, label=label.value.name, union=True),
                                                                                 categorical=label.value.kind == CATEGORICAL,
                                                                                 linear=True, svm=False, **kwargs)
                save_res()

                if dataset.frames_per_sample > 1:
                    printd("alltime input linear")
                    results[f'{label.value.name}_alltime_input_linear'] = classify_head_eval(
                        get_inp_ds(last_frame=False, pc=None, label=label.value.name, union=True),
                        categorical=label.value.kind == CATEGORICAL,
                        linear=True, svm=False, **kwargs)
                    save_res()

            if inp and f'{label.value.name}_input_pca_linear' not in results or override_linear:
                printd("input pca linear")
                if only_input:
                    if pcs:
                        for pc in pcs:
                            results[f'{label.value.name}_input_pca{pc}_linear'] = classify_head_eval(
                                get_inp_ds(last_frame=True, pc=pc, label=label.value.name, union=True),
                                categorical=label.value.kind == CATEGORICAL,
                                linear=True, svm=False, **kwargs)
                            save_res()

                else:
                    results[f'{label.value.name}_input_pca_linear'] = classify_head_eval(get_inp_ds(last_frame=True,
                                                                                                    pc=x_train_embd.shape[-2],
                                                                                                    label=label.value.name, union=True),
                                                                                     categorical=label.value.kind == CATEGORICAL,
                                                                                     linear=True, svm=False, **kwargs)
                    save_res()

                if dataset.frames_per_sample > 1:
                    printd("alltime input pca linear")
                    if only_input:
                        if pcs:
                            for pc in pcs:
                                results[f'{label.value.name}_alltime_input_pca{pc}_linear'] = classify_head_eval(
                                    get_inp_ds(last_frame=False, pc=pc, label=label.value.name, union=True),
                                    categorical=label.value.kind == CATEGORICAL,
                                    linear=True, svm=False, **kwargs)
                                save_res()
                    else:
                        results[f'{label.value.name}_alltime_input_pca_linear'] = classify_head_eval(
                            get_inp_ds(last_frame=False, pc=x_train_embd.shape[-2], label=label.value.name, union=True),
                            categorical=label.value.kind == CATEGORICAL,
                            linear=True, svm=False, **kwargs)
                        save_res()

        if ensemble:
            if not only_input and (not any(['linear' in k and 'ensemble' in k and 'alltime' not in k and label.value.name in k for k in results]) or override_linear):
                printd("ensemble linear")
                results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False,
                                                           base_name=f"{label.value.name}_",
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean], **kwargs))
                save_res()

            if not only_input and (dataset.frames_per_sample > 1):
                if not any(['linear' in k and 'ensemble' in k and 'alltime' in k and label.value.name in k for k in results]) or override_linear:
                    printd("ensemble linear alltime")
                    results.update(classify_head_eval_ensemble(embd_alltime_dataset, linear=True, svm=False,
                                                               base_name=f"{label.value.name}_alltime_",
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               voting_methods=[
                                                                   EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                               **kwargs))
                    save_res()

            if with_pred:
                if not only_input and (
                        not any(['linear' in k and 'ensemble' in k and 'nopred' in k and 'alltime' not in k and label.value.name in k for
                                 k in results]) or override_linear):
                    printd("ensemble nopred linear")
                    results.update(classify_head_eval_ensemble(embd_dataset_nopred, linear=True, svm=False,
                                                               base_name=f"{label.value.name}_nopred_",
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               voting_methods=[
                                                                   EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                               **kwargs))
                    save_res()

                if not only_input and (dataset.frames_per_sample > 1):
                    if not any(['linear' in k and 'ensemble' in k and 'nopred' in k and 'alltime' in k and label.value.name in k for k in
                                results]) or override_linear:
                        printd("ensemble linear alltime")
                        results.update(classify_head_eval_ensemble(embd_alltime_dataset_nopred, linear=True, svm=False,
                                                                   base_name=f"{label.value.name}_nopred_alltime_",
                                                                   categorical=label.value.kind == CATEGORICAL,
                                                                   voting_methods=[
                                                                       EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                                   **kwargs))
                        save_res()

            if inp and (not any([k.startswith(f"{label.value.name}_input_pathway") for k in results.keys()]) or override_linear):
                printd("ensemble input linear")
                results.update(classify_head_eval_ensemble(get_inp_ds(last_frame=True, pc=None, label=label.value.name, union=False),
                                                           base_name=f"{label.value.name}_input_", svm=False,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                save_res()

                if dataset.frames_per_sample > 1:
                    printd("ensemble input alltime linear")
                    results.update(
                        classify_head_eval_ensemble(get_inp_ds(last_frame=False, pc=None, label=label.value.name, union=False),
                                                    base_name=f"{label.value.name}_alltime_input_", svm=False,
                                                    categorical=label.value.kind == CATEGORICAL,
                                                    voting_methods=[
                                                        EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]),
                        **kwargs)
                    save_res()

            if inp and (not any([k.startswith(f"{label.value.name}_input_pca_pathway") for k in results.keys()]) or override_linear):
                printd("ensemble input pca linear")
                if only_input:
                    if pcs:
                        for pc in pcs:
                            results.update(classify_head_eval_ensemble(
                                get_inp_ds(last_frame=True, pc=pc, label=label.value.name, union=False),
                                base_name=f"{label.value.name}_input_pca{pc}_", svm=False,
                                categorical=label.value.kind == CATEGORICAL,
                                voting_methods=[
                                    EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]),
                                           **kwargs)
                            save_res()
                else:
                    results.update(classify_head_eval_ensemble(get_inp_ds(last_frame=True, pc=x_train_embd.shape[-2], label=label.value.name, union=False),
                                                               base_name=f"{label.value.name}_input_pca_", svm=False,
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                    save_res()

                if dataset.frames_per_sample > 1:
                    printd("ensemble input pca alltime linear")
                    if only_input:
                        if pcs:
                            for pc in pcs:
                                results.update(
                                    classify_head_eval_ensemble(
                                        get_inp_ds(last_frame=False, pc=pc, label=label.value.name, union=False),
                                        base_name=f"{label.value.name}_alltime_input_pca{pc}_", svm=False,
                                        categorical=label.value.kind == CATEGORICAL,
                                        voting_methods=[
                                            EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]),
                                    **kwargs)
                                save_res()
                    else:
                        results.update(
                            classify_head_eval_ensemble(get_inp_ds(last_frame=False, pc=x_train_embd.shape[-2], label=label.value.name, union=False),
                                                        base_name=f"{label.value.name}_alltime_input_pca_", svm=False,
                                                        categorical=label.value.kind == CATEGORICAL,
                                                        voting_methods=[
                                                            EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]),
                            **kwargs)
                        save_res()

        if knn:
            for k in ks:
                cur_name = f"{label.value.name}_k={k}"
                if not only_input and (cur_name not in results):
                    printd(cur_name, ":", end='\t')
                    results[cur_name] = classify_head_eval(embd_dataset,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           linear=False, k=k, **kwargs)
                    save_res()

                if not only_input and dataset.frames_per_sample > 1:
                    cur_name = f"{label.value.name}_alltime_k={k}"
                    if cur_name not in results:
                        printd(cur_name, ":", end='\t')
                        results[cur_name] = classify_head_eval(embd_alltime_dataset,
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               linear=False, k=k, **kwargs)
                        save_res()

                cur_name = f"{label.value.name}_input_k={k}"
                if inp and (cur_name not in results):
                    printd(cur_name, ":", end='\t')
                    results[cur_name] = classify_head_eval(get_inp_ds(last_frame=True, pc=None, label=label.value.name, union=False),
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           linear=False, k=k, **kwargs)
                    save_res()
                if dataset.frames_per_sample > 1:
                    cur_name = f"{label.value.name}_alltime_input_k={k}"
                    if inp and (cur_name not in results):
                        printd(cur_name, ":", end='\t')
                        results[cur_name] = classify_head_eval(get_inp_ds(last_frame=False, pc=None, label=label.value.name, union=True),
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               linear=False, k=k, **kwargs)
                        save_res()
                if only_input:
                    if pcs:
                        for pc in pcs:
                            cur_name = f"{label.value.name}_input_pca{pc}_k={k}"
                            if inp and (cur_name not in results):
                                printd(cur_name, ":", end='\t')
                                results[cur_name] = classify_head_eval(
                                    get_inp_ds(last_frame=True, pc=pc, label=label.value.name, union=True),
                                    categorical=label.value.kind == CATEGORICAL,
                                    linear=False, k=k, **kwargs)
                                save_res()
                else:
                    cur_name = f"{label.value.name}_input_pca_k={k}"
                    if inp and (cur_name not in results):
                        printd(cur_name, ":", end='\t')
                        results[cur_name] = classify_head_eval(get_inp_ds(last_frame=True, pc=x_train_embd.shape[-2], label=label.value.name, union=True),
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               linear=False, k=k, **kwargs)
                        save_res()

                if dataset.frames_per_sample > 1:
                    if only_input:
                        if pcs:
                            for pc in pcs:
                                cur_name = f"{label.value.name}_alltime_input_pca{pc}_k={k}"
                                if inp and (cur_name not in results):
                                    printd(cur_name, ":", end='\t')
                                    results[cur_name] = classify_head_eval(
                                        get_inp_ds(last_frame=False, pc=pc, label=label.value.name, union=True),
                                        categorical=label.value.kind == CATEGORICAL,
                                        linear=False, k=k, **kwargs)
                                    save_res()
                    else:
                        cur_name = f"{label.value.name}_alltime_input_pca_k={k}"
                        if inp and (cur_name not in results):
                            printd(cur_name, ":", end='\t')
                            results[cur_name] = classify_head_eval(get_inp_ds(last_frame=False, pc=x_train_embd.shape[-2], label=label.value.name, union=True),
                                                                   categorical=label.value.kind == CATEGORICAL,
                                                                   linear=False, k=k, **kwargs)
                            save_res()

        if ensemble and knn:

            for k in ks:
                if not only_input and (not any(['k=' in key and 'ensemble' in key and 'alltime' not in key and label.value.name in key for key in results])):
                    printd(f"ensemble k={k}")
                    results.update(classify_head_eval_ensemble(embd_dataset, linear=False, svm=False, k=k,
                                                               base_name=f"{label.value.name}_k={k}_",
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                    save_res()

                if dataset.frames_per_sample > 1:
                    if not only_input and (not any(['k=' in key and 'ensemble' in key and 'alltime' in key and label.value.name in key
                                                    for key in results])):
                        printd(f"ensemble alltime k={k}")
                        results.update(classify_head_eval_ensemble(embd_alltime_dataset, linear=False, svm=False, k=k,
                                                                   base_name=f"{label.value.name}_alltime_k={k}_",
                                                                   categorical=label.value.kind == CATEGORICAL,
                                                                   voting_methods=[
                                                                       EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]),
                                       **kwargs)
                        save_res()

                if inp:
                    if not any(['k=' in key and 'ensemble' in key and 'alltime' not in key and 'input' in key and label.value.name in key
                                for key in results]):
                        printd(f"input ensemble k={k}")
                        results.update(classify_head_eval_ensemble(get_inp_ds(last_frame=True, pc=None, label=label.value.name, union=False),
                                                                   base_name=f"{label.value.name}_input_k={k}_",
                                                                        svm=False, k=k, linear=False,
                                                                        categorical=label.value.kind == CATEGORICAL,
                                                                        voting_methods=[
                                                                            EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                                        **kwargs))
                        save_res()

                    if dataset.frames_per_sample > 1:
                        if not any([
                                       'k=' in key and 'ensemble' in key and 'alltime' in key and 'input' in key and label.value.name in key
                                       for key in results]):
                            printd(f"input alltime ensemble k={k}")
                            results.update(
                                classify_head_eval_ensemble(get_inp_ds(last_frame=False, pc=None, label=label.value.name, union=False),
                                                            base_name=f"{label.value.name}_alltime_input_k={k}_",
                                                            svm=False, k=k, linear=False,
                                                            categorical=label.value.kind == CATEGORICAL,
                                                            voting_methods=[
                                                                EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                            **kwargs))
                            save_res()
                if inp:
                    if not any(['k=' in key and 'ensemble' in key and 'alltime' not in key and 'input' in key and label.value.name in key and "pca" in key
                                for key in results]):
                        printd(f"input pca ensemble k={k}")
                        if only_input:
                            if pcs:
                                for pc in pcs:
                                    results.update(classify_head_eval_ensemble(
                                        get_inp_ds(last_frame=True, pc=pc, label=label.value.name, union=False),
                                        base_name=f"{label.value.name}_input_pca{pc}_k={k}_",
                                        svm=False, k=k, linear=False,
                                        categorical=label.value.kind == CATEGORICAL,
                                        voting_methods=[
                                            EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                        **kwargs))
                                    save_res()

                        else:
                            results.update(classify_head_eval_ensemble(get_inp_ds(last_frame=True, pc=x_train_embd.shape[-2], label=label.value.name, union=False),
                                                                       base_name=f"{label.value.name}_input_pca_k={k}_",
                                                                            svm=False, k=k, linear=False,
                                                                            categorical=label.value.kind == CATEGORICAL,
                                                                            voting_methods=[
                                                                                EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                                            **kwargs))
                            save_res()

                    if dataset.frames_per_sample > 1:
                        if not any([
                                       'k=' in key and 'ensemble' in key and 'alltime' in key and 'input' in key and label.value.name in key and "pca" in key
                                       for key in results]):
                            printd(f"input pca alltime ensemble k={k}")
                            if only_input:
                                if pcs:
                                    for pc in pcs:
                                        results.update(
                                            classify_head_eval_ensemble(
                                                get_inp_ds(last_frame=False, pc=pc, label=label.value.name,
                                                           union=False),
                                                base_name=f"{label.value.name}_alltime_input_pca{pc}_k={k}_",
                                                svm=False, k=k, linear=False,
                                                categorical=label.value.kind == CATEGORICAL,
                                                voting_methods=[
                                                    EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                **kwargs))
                                        save_res()
                            else:
                                results.update(
                                    classify_head_eval_ensemble(get_inp_ds(last_frame=False, pc=x_train_embd.shape[-2], label=label.value.name, union=False),
                                                                base_name=f"{label.value.name}_alltime_input_pca_k={k}_",
                                                                svm=False, k=k, linear=False,
                                                                categorical=label.value.kind == CATEGORICAL,
                                                                voting_methods=[
                                                                    EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                                **kwargs))
                                save_res()

    if (predict and not any(['predict' in k for k in results])) or override_predict:

        if with_pred:
            print("predict with embedding")
            evaluate_predict(results,
                             get_inp_ds(last_frame=False, pc=None, label=None, union=False, flatten=False),
                             get_inp_ds(last_frame=False, pc=None, label=None, union=True, flatten=False),
                             None if only_input else Data(x_train_embd[..., :-bins_per_frame, :x_train_embd.shape[-2]//2, :], None,
                                                          x_test_embd[..., :-bins_per_frame, :x_test_embd.shape[-2]//2, :], None,
                                                          x_val=x_val_embd[..., :-bins_per_frame, :x_val_embd.shape[-2]//2, :], y_val=None,
                                                          flatten_y=False, normalize=True),
                             encoder_removed_bins, bins_per_frame=1, run_inp=inp, base='embd')
            save_res()

            print("predict with predictor")
            evaluate_predict(results,
                             get_inp_ds(last_frame=False, pc=None, label=None, union=False,flatten=False),
                             get_inp_ds(last_frame=False, pc=None, label=None, union=True,flatten=False),
                             None if only_input else Data(x_train_embd[..., x_train_embd.shape[-2]//2:, :], None,
                                                          x_test_embd[..., x_test_embd.shape[-2]//2:, :], None,
                                                          x_val=x_val_embd[..., x_val_embd.shape[-2]//2:, :], y_val=None,
                                                          flatten_y=False, normalize=True),
                             encoder_removed_bins, bins_per_frame=1, run_inp=inp, base='predictor')
            save_res()

        print("predict with output")
        evaluate_predict(results,
                         get_inp_ds(last_frame=False, pc=None, label=None, union=False, flatten=False),
                         get_inp_ds(last_frame=False, pc=None, label=None, union=True, flatten=False),
                         None if only_input else Data(x_train_embd[:, :-bins_per_frame], None,
                                                      x_test_embd[:, :-bins_per_frame], None,
                                                      x_val=x_val_embd[:, :-bins_per_frame], y_val=None,
                                                      flatten_y=False, normalize=True),
                         encoder_removed_bins, bins_per_frame=1, run_inp=inp)
        save_res()
    return results
