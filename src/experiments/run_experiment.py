from sklearnex import patch_sklearn
patch_sklearn()
from torch import multiprocessing as mp
from tqdm.auto import tqdm
import os
from src.experiments.experiment_loader import load_experiment_cfg
from src.experiments.utils import get_result_dir, apply_transforms, get_data_manager, export_results_df

def run_experiment(experiment, model_id, subject, fold_idx, device):
    cfg = load_experiment_cfg(experiment)
    model_cfg = cfg['models'][model_id]

    data_manager = get_data_manager(cfg, model_id, subject, fold_idx, device)
    
    opt_params = model_cfg['search_method'](experiment, subject, fold_idx, model_cfg, data_manager, device)
    model = model_cfg['class'](**model_cfg['arch_config'], device=device)
    trainer = model.get_trainer(**model_cfg['trainer_config'], **opt_params)

    transforms = {k: v['method'](**v.get('params', {})) for k, v in  opt_params.get('transforms', {}).items()}
    train_data = data_manager.get_train_val_data()
    train_data = apply_transforms(train_data, transforms, fit=True)
    trainer.fit(train_data)
    
    result_dir = get_result_dir(experiment, subject, fold_idx, model_cfg)
    trainer.save_train_data(result_dir)
    
    confounded_test_data = data_manager.get_confounded_test_data()
    confounded_test_data = apply_transforms(confounded_test_data, transforms)
    labels, logits, loss = trainer.predict_proba(confounded_test_data)
    trainer.save_test_data(result_dir, 'confounded_test', labels, logits, loss)
    tqdm.write(f'Confounded test loss: {loss}')

    unconfounded_test_data = data_manager.get_unconfounded_test_data()
    unconfounded_test_data = apply_transforms(unconfounded_test_data, transforms)
    labels, logits, loss = trainer.predict_proba(unconfounded_test_data)
    trainer.save_test_data(result_dir, 'unconfounded_test', labels, logits, loss)
    tqdm.write(f'Unconfounded test loss: {loss}')

def assign_experiment(experiment, model_id, subject, fold_idx, pool):
    device = pool.get()
    run_experiment(experiment, model_id, subject, fold_idx, device)
    pool.put(device)

def generate_experiment_list(experiments):
    complete, gpu_jobs, cpu_jobs = [], [], []
    for exp in experiments:
        cfg = load_experiment_cfg(exp)
        for model_id in tqdm(cfg['models'], desc="Evaluating models", unit="model", leave=False):
            cpu_only = cfg['models'][model_id].get('cpu_only', False)
            for subject in tqdm(cfg['dataset']['subjects'], desc="Evaluating subjects", unit="subject", leave=False):
                for fold_idx in range(cfg['scheme']['n_folds']):
                    result_dir = get_result_dir(exp, subject, fold_idx, cfg['models'][model_id])
                    if os.path.isdir(result_dir):
                        complete.append((exp, model_id, subject, fold_idx, result_dir))
                    else:
                        if cpu_only:
                            cpu_jobs.append((exp, model_id, subject, fold_idx, result_dir))
                        else:
                            gpu_jobs.append((exp, model_id, subject, fold_idx, result_dir))
    return complete, gpu_jobs, cpu_jobs

def run(experiments, devices):

    pool = mp.Queue()
    for device in devices:
        pool.put(device)
    processes = []

    complete, gpu_jobs, cpu_jobs = generate_experiment_list(experiments)
    for experiment, model_id, subject, fold_idx, _ in gpu_jobs:
        process = mp.Process(target=assign_experiment,
                            args=(experiment, model_id, subject, fold_idx, pool))
        processes.append(process)
        process.start()

    with tqdm(total=len(complete) + len(cpu_jobs) + len(gpu_jobs), initial=len(complete), desc="Running experiments", unit="experiment") as pbar:
        for process in processes:
            process.join()
            pbar.update(1)
        
        for experiment, model_id, subject, fold_idx, _ in cpu_jobs:
            run_experiment(experiment, model_id, subject, fold_idx, 'cpu')
            pbar.update(1)

    return export_results_df(experiments, complete + gpu_jobs + cpu_jobs)