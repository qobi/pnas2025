import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from src.experiments.experiment_loader import load_experiment_cfg
from src.experiments.utils import get_result_path, get_trained_model_dir

def extract_fold_results(scheme_cfg, subject, fold_idx, model, categories, pseudocategories, exemplars, predictions, hyperparameters, trial_idx):
    results = {'task': [scheme_cfg['target'] + '_decoding']*len(trial_idx),
               'hyperparameter_tuning_scheme': [scheme_cfg['hyperparameter_tuning_scheme']]*len(trial_idx),
               'cross_validation_scheme': [scheme_cfg['cross_validation_scheme']]*len(trial_idx),
               'model': [model]*len(trial_idx),
               'subject': [subject]*len(trial_idx), 
               'fold_idx': [fold_idx]*len(trial_idx),
               'hyperparameters': [hyperparameters]*len(trial_idx),
               'category': categories[trial_idx],
               'pseudocategory': pseudocategories[trial_idx],
               'exemplar': exemplars[trial_idx],
               'predicted': predictions,
               'trial_idx': trial_idx}
    return results

def build_results_df(experiment_IDs):

    results = {'task': [],
               'hyperparameter_tuning_scheme': [],
               'cross_validation_scheme': [],
               'model': [],
               'subject': [], 
               'fold_idx': [],
               'hyperparameters': [],
               'category': [],
               'pseudocategory': [],
               'exemplar': [], 
               'predicted': [],
               'trial_idx': []}

    for experiment_ID in experiment_IDs:
        experiment_cfg = load_experiment_cfg(experiment_ID)
        dataset_cfg = experiment_cfg['dataset']
        scheme_cfg = experiment_cfg['scheme']
        for subject in experiment_cfg['dataset']['subjects']:
            data_dir = os.path.join('data', dataset_cfg['name'], 'processed', subject)

            categories = torch.load(os.path.join(data_dir, 'category.pt'), weights_only=True, map_location='cpu').numpy()
            pseudocategories = torch.load(os.path.join(data_dir, 'pseudocategory.pt'), weights_only=True, map_location='cpu').numpy()
            exemplars = torch.load(os.path.join(data_dir, 'exemplar.pt'), weights_only=True, map_location='cpu').numpy()

            for fold_idx in range(experiment_cfg['scheme']['n_folds']):
                for model in experiment_cfg['models']:
                    try:
                        results_path = get_result_path(scheme_cfg, model['name'], experiment_cfg['dataset']['name'], subject, fold_idx)
                        results_dir = os.path.dirname(results_path)

                        trial_idx = torch.load(os.path.join(results_dir, 'test_idx.pth'), weights_only=True, map_location='cpu').numpy()
                        predictions = torch.load(os.path.join(results_dir, 'predictions.pth'), weights_only=True, map_location='cpu').numpy()

                        model_dir = get_trained_model_dir(scheme_cfg, model['name'], experiment_cfg['dataset']['name'], subject, fold_idx)
                        hyperparameters = torch.load(os.path.join(model_dir, 'hyperparameters.pth'), weights_only=True, map_location='cpu')
                        fold_results = extract_fold_results(scheme_cfg, 
                                                            subject, fold_idx, model['name'], 
                                                            categories, pseudocategories, exemplars, 
                                                            predictions, hyperparameters, trial_idx)
                        results.update({key: np.concatenate([results[key], fold_results[key]]) for key in results.keys()})
                    except FileNotFoundError:
                        continue

    results_df = pd.DataFrame(results)
    return results_df

def plot_accuracy_by_experiment_model_and_fold(results_df):
    results_df['correct'] = results_df['category'] == results_df['predicted']
    accuracy_df = results_df.groupby(['task', 
                                      'hyperparameter_tuning_scheme', 
                                      'cross_validation_scheme', 
                                      'model', 
                                      'subject', 
                                      'fold_idx',
                                      'category']).agg(accuracy = ('correct', 'mean')).reset_index()
    print(accuracy_df.groupby(['task', 'hyperparameter_tuning_scheme', 'cross_validation_scheme', 'model']).agg(accuracy = ('accuracy', 'mean')))
    accuracy_df['experiment_idx'] = accuracy_df.groupby(['task', 'hyperparameter_tuning_scheme', 'cross_validation_scheme', 'model', 'category']).ngroup()
    accuracy_df['experiment_name'] = accuracy_df['model'] + '-' + accuracy_df['hyperparameter_tuning_scheme'] + '-' + accuracy_df['cross_validation_scheme']
    # experiment_order = [f"{model}-{hp}-{cv}" for model in accuracy_df['model'].unique() for hp in accuracy_df['hyperparameter_tuning_scheme'].unique() for cv in accuracy_df['cross_validation_scheme'].unique()]
    experiment_order = accuracy_df['experiment_name'].sort_values().unique()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.violinplot(x='experiment_name', y='accuracy', data=accuracy_df, hue='cross_validation_scheme', ax=ax, palette='Paired', order=experiment_order)
    ax.set_title('Accuracy by Experiment and Model')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Model')
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

experiment_IDs = ["SUDB_subject_dependent_grouped_hyperparameter_tuning_grouped_cross_validation_category_decoding",
                  "SUDB_subject_dependent_stratified_hyperparameter_tuning_stratified_cross_validation_category_decoding"]
                  
results_df = build_results_df(experiment_IDs)

plot_accuracy_by_experiment_model_and_fold(results_df)

# images = [plt.imread('figures/stimuli/reformatted/stimulus{}.png'.format(i)) for i in range(1, 73)]

# # For each value of fold_idx, print the exemplars

# results_df = results_df[results_df['model'] == 'LDA']
# for cv in results_df['cross_validation_scheme'].unique():
#     fig, ax = plt.subplots(72, 12, figsize=(72, 12))
#     cv_df = results_df[results_df['cross_validation_scheme'] == cv]
#     for fold_idx in cv_df['fold_idx'].astype(int).unique():
#         for stimulus in cv_df[cv_df['fold_idx'] == fold_idx]['exemplar'].astype(int).unique():
#             # Show im with no axis and fill the corresponding subplot
#             ax[fold_idx*6 + stimulus//12, stimulus%12].imshow(images[stimulus])

    
#     # Disable axis for all subplots and add a thick line every 6 rows
#     for i in range(72):
#         if i % 6 == 0:
#             for j in range(12):
#                 ax[i, j].axhline(0, color='black', linewidth=2)
#         for j in range(12):
#             # set ax to tight
#             ax[i, j].axis('tight')
#             ax[i, j].axis('off')

#     plt.title(cv)
#     plt.tight_layout()
#     plt.show()

