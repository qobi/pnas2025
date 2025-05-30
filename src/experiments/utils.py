import os
from hashlib import sha256
from base64 import b32encode
import pandas as pd
from h5py import File
from src.data.data_manager import DataManager
from src.data.feature_engineering.utils import load_feature
from src.model_selection import load_cv_folds

def get_result_dir(experiment, subject, fold_idx, model_cfg):
    model_cfg = model_cfg.copy()
    model_id = model_cfg['id']
    model_cfg['search_method'] = model_cfg['search_method'].__name__  
    key = b32encode(sha256(str(model_cfg).encode()).digest()).decode('utf-8').rstrip('=')
    
    path = os.path.join('outputs', experiment, subject, f'fold_{fold_idx}', 'models', model_id, key)
    return path

def get_sub_result_dir(experiment, subject, fold_idx, nested_fold_idx, model_id, hyperparameters):
    key = b32encode(sha256(str(hyperparameters).encode()).digest()).decode('utf-8').rstrip('=')

    path = os.path.join('outputs', experiment, subject, f'fold_{fold_idx}', 'nested_folds', f'nested_fold_{nested_fold_idx}', model_id, key)
    return path

def is_complete(experiment, subject, fold_idx, model_cfg):
    result_dir = get_result_dir(experiment, subject, fold_idx, model_cfg)
    return os.path.isdir(result_dir)

def get_data_manager(cfg, model_id, subject, fold_idx, device):
    features = cfg['models'][model_id]['features']
    target = cfg['scheme']['target']

    features = {f: load_feature(subject, f, device) for f in features}
    features['target'] = load_feature(subject, target, device)
    folds = load_cv_folds(subject, target, device)
    
    data_manager = DataManager(features, folds)
    data_manager.set_fold_idx(fold_idx)
    return data_manager

def apply_transforms(data, transforms, fit=False):
    if fit:
        out = {k: transforms[k].fit_transform(v) if k in transforms else v for k, v in data.items()}
    else:
        out = {k: transforms[k].transform(v) if k in transforms else v for k, v in data.items()}

    return out

def export_results_df(experiments, experiment_list):
    dfs = {}
    for experiment in experiments:
        results = {
        'model': [],
        'subject': [],
        'fold_idx': [],
        'hyperparameters': [],
        'partition': [],
        'labels': [],}
        for exp, model_id, subject, fold_idx, results_dir in experiment_list:
            if exp != experiment:
                continue
            
            if not os.path.isdir(results_dir):
                raise FileNotFoundError(f"Results directory {results_dir} does not exist.")
            
            with File(os.path.join(results_dir, 'data.hdf5'), 'r') as f:
                for partition in ['confounded_test', 'unconfounded_test']:
                    logits = f[partition]['logits'][:]
                    labels = f[partition]['labels'][:]
                    hyperparameters = f[partition].attrs.get('hyperparameters', {})
             
                    results['model'].extend([model_id] * len(labels))
                    results['subject'].extend([subject] * len(labels))
                    results['fold_idx'].extend([fold_idx] * len(labels))
                    results['hyperparameters'].extend([hyperparameters] * len(labels))
                    results['partition'].extend([partition] * len(labels))
                    results['labels'].extend(labels)
                    for i in range(logits.shape[1]):
                        results[f'logit_{i}'] = results.get(f'logit_{i}', [])
                        results[f'logit_{i}'].extend(logits[:, i])
        results = pd.DataFrame(results)
        results.to_csv(os.path.join('outputs', experiment, 'results.csv'), index=False)
        dfs[experiment] = results
    return dfs
        