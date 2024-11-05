from typing import Optional

from neuronal.utils.data import Labels, CATEGORICAL
from utils.data import Data
from utils.evaluation.ensemble import EnsembleVotingMethods
from utils.evaluation.evaluation import classify_head_eval_ensemble
from utils.model.model import load_model_from_json
from utils.modules import Modules
from utils.utils import printd
import numpy as np


def get_masked_ds(model, dataset, union=False, bins_per_frame=1, last_frame=True):
    if isinstance(model, str):
        model = load_model_from_json(model, Modules.NEURONAL)
    aug_layer = model.get_layer("data_augmentation")
    pathways = model.get_layer('pathways')
    pathway_indices = pathways.indices.numpy()
    if union:
        union = np.unique(pathway_indices) - model.get_layer("pathways").shift
        setup_func = lambda x: aug_layer(x).numpy()[:, union, ..., -bins_per_frame if last_frame else 0:]
    else:
        setup_func = lambda x: np.transpose(aug_layer(x).numpy()[:, pathway_indices - pathways.shift, ..., -bins_per_frame if last_frame else 0:], [0, 1, 3, 2]).reshape(
            x.shape[0], -1, pathway_indices.shape[-1])

    ds = Data(setup_func(dataset.get_x_train()), dataset.get_y_train(),
              setup_func(dataset.get_x_test()), dataset.get_y_test(),
              x_val=setup_func(dataset.get_x_val()), y_val=dataset.get_y_val())
    return ds


def evaluate(model, dataset="SessionDataGenerator", module: Modules=Modules.NEURONAL, labels=[Labels.STIMULUS],
             knn=False, linear=True, ensemble=True, save_results=False, override=False, override_linear=False, inp=True,
             ks=[1] + list(range(5, 21, 5)), **kwargs):

    if isinstance(model, str):
        model_kwargs = module.load_json(model, config=True)
        assert model_kwargs is not None
        printd("loading model...", end='\t')
        model = load_model_from_json(model, module)
        printd("done")
        if dataset is None:
            printd("loading dataset...", end='\t')
            dataset = module.get_class_from_data(model_kwargs['dataset'])(**model_kwargs.get('data_kwargs', {}))
            printd("done")
    else:
        model_kwargs = module.load_json(model.name, config=True)


    bins_per_frame = dataset.bins_per_frame
    with_pred = "predictions" in [l.name for l in model.layers]

    def transform_embedding(embedding, last_frame=True):
        normalize = lambda arr: arr/np.linalg.norm(arr, keepdims=True, axis=-2) if model_kwargs['loss_kwargs'].pop("cosine", False) else arr

        if last_frame:
            encoder_removed_bins = model.get_layer("pathways").output_shape[-1] != model.get_layer("embedding").output_shape[1]
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
    x_train_pred = model.predict(dataset.get_x_train())[0]
    x_test_pred = model.predict(dataset.get_x_test())[0]
    x_train_embd = transform_embedding(x_train_pred)
    x_train_embd_alltime = transform_embedding(x_train_pred, last_frame=False)

    x_test_embd = transform_embedding(x_test_pred)
    x_test_embd_alltime = transform_embedding(x_test_pred, last_frame=False)

    x_val = dataset.get_x_val()
    if x_val is not None:
        x_val_pred = model.predict(x_val)[0]
        x_val_embd = transform_embedding(x_val_pred)
        x_val_embd_alltime = transform_embedding(x_val_pred, last_frame=False)
    else:
        x_val_embd = None
        x_val_embd_alltime = None

    from sklearn.decomposition import PCA
    pca = PCA(x_train_embd.shape[-2])
    pca.fit(np.concatenate([dataset.get_x_train()[..., i] for i in range(dataset.get_x_train().shape[-1])], axis=0))
    x_train_pca = np.stack([pca.transform(dataset.get_x_train()[..., i]) for i in range(dataset.get_x_train().shape[-1])], axis=-1)
    x_test_pca = np.stack([pca.transform(dataset.get_x_test()[..., i]) for i in range(dataset.get_x_test().shape[-1])], axis=-1)
    if x_val is not None:
        x_val_pca = np.stack(
            [pca.transform(dataset.get_x_val()[..., i]) for i in range(dataset.get_x_val().shape[-1])], axis=-1)
    else:
        x_val_pca = None

    y_train = dataset.get_y_train(labels)
    y_test = dataset.get_y_test(labels)
    y_val = dataset.get_y_val(labels)

    printd("done")

    results = module.load_evaluation_json(model.name) if not override else {}

    if results is None:
        results = {}

    save_res = lambda *inputs: module.save_evaluation_json(model.name, results) if save_results else None

    for label in labels:
        printd(f"evaluating label {label.value.name}")
        basic_dataset = Data(dataset.get_x_train(), y_train[label.value.name],
                             dataset.get_x_test(), y_test[label.value.name],
                             x_val=dataset.get_x_val(), y_val=y_val[label.value.name] if y_val is not None else None,
                             normalize=True)

        basic_pca_dataset = Data(x_train_pca, basic_dataset.get_y_train(),
                                 x_test_pca, basic_dataset.get_y_test(),
                                 x_val=x_val_pca, y_val=y_val[label.value.name] if y_val is not None else None
                                 )

        embd_dataset = Data(x_train_embd, y_train[label.value.name],
                            x_test_embd, y_test[label.value.name],
                            x_val=x_val_embd, y_val=y_val[label.value.name] if y_val is not None else None,
                            normalize=True)
        embd_alltime_dataset = Data(x_train_embd_alltime, y_train[label.value.name],
                                    x_test_embd_alltime, y_test[label.value.name],
                                    x_val=x_val_embd_alltime, y_val=y_val[label.value.name] if y_val is not None else None,
                                    normalize=True)

        if with_pred:
            remove_pred = lambda embd: embd[:, :embd.shape[-3]//2]
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
            if f'{label.value.name}_linear' not in results or override_linear:
                printd("linear")
                results[f'{label.value.name}_linear'] = classify_head_eval(embd_dataset,
                                                                           categorical=label.value.kind == CATEGORICAL,
                                                                           linear=True, svm=False, **kwargs)
                save_res()

            if with_pred and f'{label.value.name}_nopred_linear' not in results or override_linear:
                printd("nopred linear")
                results[f'{label.value.name}_nopred_linear'] = classify_head_eval(embd_dataset_nopred,
                                                                                  categorical=label.value.kind == CATEGORICAL,
                                                                                  linear=True, svm=False, **kwargs)
                save_res()

            if dataset.frames_per_sample > 1:
                if f'{label.value.name}_alltime_linear' not in results or override_linear:
                    printd("alltime linear")
                    results[f'{label.value.name}_alltime_linear'] = classify_head_eval(embd_alltime_dataset,
                                                                                       categorical=label.value.kind == CATEGORICAL,
                                                                                       linear=True, svm=False, **kwargs)
                    save_res()

            if with_pred and f'{label.value.name}_nopred_alltime_linear' not in results or override_linear:
                printd("nopred alltime linear")
                results[f'{label.value.name}_nopred_alltime_linear'] = classify_head_eval(embd_alltime_dataset_nopred,
                                                                                  categorical=label.value.kind == CATEGORICAL,
                                                                                  linear=True, svm=False, **kwargs)
                save_res()

            if inp and (f'{label.value.name}_input_linear' not in results or override_linear):
                printd("input linear")
                results[f'{label.value.name}_input_linear'] = classify_head_eval(get_masked_ds(model, dataset=basic_dataset,
                                                                                               bins_per_frame=dataset.bins_per_frame, union=True),
                                                                                 categorical=label.value.kind == CATEGORICAL,
                                                                                 linear=True, svm=False, **kwargs)
                save_res()

                if dataset.frames_per_sample > 1:
                    printd("alltime input linear")
                    results[f'{label.value.name}_alltime_input_linear'] = classify_head_eval(
                        get_masked_ds(model, dataset=basic_dataset, union=True, last_frame=False),
                        categorical=label.value.kind == CATEGORICAL,
                        linear=True, svm=False, **kwargs)
                    save_res()

            if inp and f'{label.value.name}_input_pca_linear' not in results or override_linear:
                printd("input pca linear")
                results[f'{label.value.name}_input_pca_linear'] = classify_head_eval(get_masked_ds(model, dataset=basic_pca_dataset,
                                                                                               bins_per_frame=dataset.bins_per_frame, union=True),
                                                                                 categorical=label.value.kind == CATEGORICAL,
                                                                                 linear=True, svm=False, **kwargs)
                save_res()

                if dataset.frames_per_sample > 1:
                    printd("alltime input pca linear")
                    results[f'{label.value.name}_alltime_input_pca_linear'] = classify_head_eval(
                        get_masked_ds(model, dataset=basic_pca_dataset, union=True, last_frame=False),
                        categorical=label.value.kind == CATEGORICAL,
                        linear=True, svm=False, **kwargs)
                    save_res()

        if ensemble:
            if not any(['linear' in k and 'ensemble' in k and 'alltime' not in k and label.value.name in k for k in results]) or override_linear:
                printd("ensemble linear")
                results.update(classify_head_eval_ensemble(embd_dataset, linear=True, svm=False,
                                                           base_name=f"{label.value.name}_",
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean], **kwargs))
                save_res()

            if dataset.frames_per_sample > 1:
                if not any(['linear' in k and 'ensemble' in k and 'alltime' in k and label.value.name in k for k in results]) or override_linear:
                    printd("ensemble linear alltime")
                    results.update(classify_head_eval_ensemble(embd_alltime_dataset, linear=True, svm=False,
                                                               base_name=f"{label.value.name}_alltime",
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               voting_methods=[
                                                                   EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                               **kwargs))
                    save_res()

            if inp and (not any([k.startswith(f"{label.value.name}_input_pathway") for k in results.keys()]) or override_linear):
                printd("ensemble input linear")
                masked_ds = get_masked_ds(model, dataset=basic_dataset, bins_per_frame=dataset.bins_per_frame)
                results.update(classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_input_", svm=False,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                save_res()

                if dataset.frames_per_sample > 1:
                    printd("ensemble input alltime linear")
                    results.update(
                        classify_head_eval_ensemble(get_masked_ds(model, dataset=basic_dataset,
                                                                  bins_per_frame=dataset.bins_per_frame, last_frame=False),
                                                    base_name=f"{label.value.name}_alltime_input_", svm=False,
                                                    categorical=label.value.kind == CATEGORICAL,
                                                    voting_methods=[
                                                        EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]),
                        **kwargs)
                    save_res()

            if inp and (not any([k.startswith(f"{label.value.name}_input_pca_pathway") for k in results.keys()]) or override_linear):
                printd("ensemble input pca linear")
                masked_ds = get_masked_ds(model, dataset=basic_pca_dataset, bins_per_frame=dataset.bins_per_frame)
                results.update(classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_input_pca_", svm=False,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                save_res()

                if dataset.frames_per_sample > 1:
                    printd("ensemble input pca alltime linear")
                    results.update(
                        classify_head_eval_ensemble(get_masked_ds(model, dataset=basic_pca_dataset,
                                                                  bins_per_frame=dataset.bins_per_frame, last_frame=False),
                                                    base_name=f"{label.value.name}_alltime_input_pca_", svm=False,
                                                    categorical=label.value.kind == CATEGORICAL,
                                                    voting_methods=[
                                                        EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]),
                        **kwargs)
                    save_res()

        if knn:
            for k in ks:
                cur_name = f"{label.value.name}_k={k}"
                if cur_name not in results:
                    printd(cur_name, ":", end='\t')
                    results[cur_name] = classify_head_eval(embd_dataset,
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           linear=False, k=k, **kwargs)
                    save_res()

                if dataset.frames_per_sample > 1:
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
                    results[cur_name] = classify_head_eval(get_masked_ds(model, dataset=basic_dataset,
                                                                         bins_per_frame=dataset.bins_per_frame, union=True),
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           linear=False, k=k, **kwargs)
                    save_res()
                if dataset.frames_per_sample > 1:
                    cur_name = f"{label.value.name}_alltime_input_k={k}"
                    if inp and (cur_name not in results):
                        printd(cur_name, ":", end='\t')
                        results[cur_name] = classify_head_eval(get_masked_ds(model, dataset=basic_dataset,
                                                                             last_frame=False,
                                                                             bins_per_frame=dataset.bins_per_frame, union=True),
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               linear=False, k=k, **kwargs)
                        save_res()

                cur_name = f"{label.value.name}_input_pca_k={k}"
                if inp and (cur_name not in results):
                    printd(cur_name, ":", end='\t')
                    results[cur_name] = classify_head_eval(get_masked_ds(model, dataset=basic_pca_dataset,
                                                                         bins_per_frame=dataset.bins_per_frame, union=True),
                                                           categorical=label.value.kind == CATEGORICAL,
                                                           linear=False, k=k, **kwargs)
                    save_res()
                if dataset.frames_per_sample > 1:
                    cur_name = f"{label.value.name}_alltime_input_pca_k={k}"
                    if inp and (cur_name not in results):
                        printd(cur_name, ":", end='\t')
                        results[cur_name] = classify_head_eval(get_masked_ds(model, dataset=basic_pca_dataset,
                                                                             last_frame=False,
                                                                             bins_per_frame=dataset.bins_per_frame, union=True),
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               linear=False, k=k, **kwargs)
                        save_res()

        if ensemble and knn:
            for k in ks:
                if not any(['k=' in key and 'ensemble' in key and 'alltime' not in key and label.value.name in key for key in results]):
                    printd(f"ensemble k={k}")
                    results.update(classify_head_eval_ensemble(embd_dataset, linear=False, svm=False, k=k,
                                                               base_name=f"{label.value.name}_k={k}_",
                                                               categorical=label.value.kind == CATEGORICAL,
                                                               voting_methods=[EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]), **kwargs)
                    save_res()

                if dataset.frames_per_sample > 1:
                    if not any(['k=' in key and 'ensemble' in key and 'alltime' in key and label.value.name in key
                                for key in results]):
                        printd(f"ensemble alltime k={k}")
                        results.update(classify_head_eval_ensemble(embd_alltime_dataset, linear=False, svm=False, k=k,
                                                                   base_name=f"{label.value.name}_alltime_k={k}_",
                                                                   categorical=label.value.kind == CATEGORICAL,
                                                                   voting_methods=[
                                                                       EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean]),
                                       **kwargs)
                        save_res()

                if inp:
                    masked_ds = get_masked_ds(model, dataset=basic_dataset, bins_per_frame=bins_per_frame)

                    if not any(['k=' in key and 'ensemble' in key and 'alltime' not in key and 'input' in key and label.value.name in key
                                for key in results]):
                        printd(f"input ensemble k={k}")
                        results.update(classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_input_k={k}_",
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
                            masked_ds = get_masked_ds(model, dataset=basic_dataset,
                                                      bins_per_frame=bins_per_frame, last_frame=False)
                            results.update(
                                classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_alltime_input_k={k}_",
                                                            svm=False, k=k, linear=False,
                                                            categorical=label.value.kind == CATEGORICAL,
                                                            voting_methods=[
                                                                EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                            **kwargs))
                            save_res()
                if inp:
                    masked_ds = get_masked_ds(model, dataset=basic_pca_dataset, bins_per_frame=bins_per_frame)

                    if not any(['k=' in key and 'ensemble' in key and 'alltime' not in key and 'input' in key and label.value.name in key and "pca" in key
                                for key in results]):
                        printd(f"input pca ensemble k={k}")
                        results.update(classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_input_pca_k={k}_",
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
                            masked_ds = get_masked_ds(model, dataset=basic_pca_dataset,
                                                      bins_per_frame=bins_per_frame, last_frame=False)
                            results.update(
                                classify_head_eval_ensemble(masked_ds, base_name=f"{label.value.name}_alltime_input_pca_k={k}_",
                                                            svm=False, k=k, linear=False,
                                                            categorical=label.value.kind == CATEGORICAL,
                                                            voting_methods=[
                                                                EnsembleVotingMethods.ArgmaxMeanProb if label.value.kind == CATEGORICAL else EnsembleVotingMethods.Mean],
                                                            **kwargs))
                            save_res()
    return results
