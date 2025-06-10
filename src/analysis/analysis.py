import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_1samp
from statsmodels.stats.multitest import multipletests
from pymer4.models import Lmer
from tqdm.auto import tqdm
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{tgheros}%\
                                \usepackage{sansmath}%\
                                \sansmath')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "tgheros"

MODELS = {'primary': [
            'LDA',
            'ADCNN',
            'AW1DCNN',
            'EEGCT-Slim',
            'EEGCT-Wide',
            'RLSTM',
            'STST',
            'TSCNN'
            ],
          'additional': [
            'LR',
            'kNN',
            'SVC',
            'ShallowConvNet',
            'DeepConvNet',
            'EEGNet'
            ]
          }

CATEGORY_MAP = {0: 'Human Body', 1: 'Human Face', 2: 'Animal Body', 3: 'Animal Face', 4: 'Natural Object', 5: 'Artificial Object'}
CATEGORY_ABBR_MAP = {0: 'HB', 1: 'HF', 2: 'AB', 3: 'AF', 4: 'NO', 5: 'AO'}

def get_predictions(df):
    logits_cols = [col for col in df.columns if col.startswith('logit_')]
    df['target_pred'] = df[logits_cols].idxmax(axis=1).str.replace('logit_', '').astype(int)
    df['category'] = df['labels'].map(CATEGORY_MAP)
    df['category_pred'] = df['target_pred'].map(CATEGORY_MAP)
    df['correct'] = (df['target_pred'] == df['labels']).astype(int)
    return df

def get_fold_level_results(df):
    df = df.groupby(['model', 'subject', 'fold_idx', 'partition']).agg(accuracy=('correct', 'mean')).reset_index()
    df['accuracy'] = df['accuracy'] * 100
    df = df.pivot_table(index=['model', 'subject', 'fold_idx'],
                        columns='partition',
                        values='accuracy').reset_index()
    df['bias'] = df['confounded_test'] - df['unconfounded_test']
    return df

def get_subject_level_results(df):
    df = get_fold_level_results(df)
    df = df.groupby(['model', 'subject']).agg(
        confounded_test=('confounded_test', 'mean'),
        confounded_test_std=('confounded_test', 'std'),
        unconfounded_test=('unconfounded_test', 'mean'),
        unconfounded_test_std=('unconfounded_test', 'std'),
        bias=('bias', 'mean'),
        bias_std=('bias', 'std')
    ).reset_index()
    return df

def get_model_level_results(df):
    df = get_subject_level_results(df)
    df = df.groupby('model').agg(
        confounded_test=('confounded_test', 'mean'),
        confounded_test_std=('confounded_test_std', 'std'),
        unconfounded_test=('unconfounded_test', 'mean'),
        unconfounded_test_std=('unconfounded_test_std', 'std'),
        bias=('bias', 'mean'),
        bias_std=('bias_std', 'std')
    ).reset_index()
    return df

def get_subject_level_category_results(df):
    df = df.groupby(['model', 'subject', 'partition', 'labels']).agg(accuracy=('correct', 'mean')).reset_index()
    df['accuracy'] = df['accuracy'] * 100
    df = df.pivot_table(index=['model', 'subject', 'labels'],
                        columns='partition',
                        values='accuracy').reset_index()
    df['category'] = df['labels'].map(CATEGORY_MAP)
    df['bias'] = df['confounded_test'] - df['unconfounded_test']
    return df

def plot_bias_boxplot_by_model_and_subject(df, model_type='primary', path=None):

    bias_df = get_fold_level_results(df)

    bias_df = bias_df[bias_df['model'].isin(MODELS[model_type])]
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.grid(which='both', axis='y', linestyle='-', linewidth=0.5)

    ref_line = ax.axhline(0, color='black', linewidth=2, linestyle='--', zorder=2)


    sns.boxplot(ax=ax, data=bias_df, x='model', y='bias', hue='subject', dodge=True,
                order=MODELS[model_type], hue_order=[f"S{i}" for i in range(1, 11)],
                legend=True,
    )

    handles, labels = ax.get_legend_handles_labels()
    subject_legend = fig.legend(title="Subject", ncol=10,
                                    handles=handles, labels=labels,
                                    loc='lower center', bbox_to_anchor=(0.535, 1.01), borderaxespad=0.,
                                    fontsize=14, title_fontsize='x-large',
                                    )
    ref_legend = fig.legend(title='Reference', loc='upper left', fontsize=14, title_fontsize='x-large',
        handles=[ref_line], labels=['No bias'], bbox_to_anchor=(0.075, 0.99))

    lower_bound, upper_bound = bias_df['bias'].min(), bias_df['bias'].max()
    lower_bound, upper_bound = int(np.floor(lower_bound / 5 +1))* 5, int(np.ceil(upper_bound / 5)) * 5

    ax.set_ylabel('Absolute bias (\%)', fontsize=18, labelpad=10)
    ax.set_yticks(np.arange(lower_bound, upper_bound, 5), ['{:d}\%'.format(x) for x in np.arange(lower_bound, upper_bound, 5)])

    ax.set_xticklabels([f'\\Large {x}' for x in MODELS[model_type]])
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel("")
    ax.set_xlabel('Decoding model', fontsize=18, labelpad=10)

    ax.get_legend().remove()

    if path:
        plt.savefig(path, bbox_inches='tight', format='pdf')
    else:
        plt.show()

def test_bias_significance(df, alpha=0.05):
    bias_df = get_subject_level_results(df)

    bias_test_df = []
    for model in MODELS['primary'] + MODELS['additional']:
        model_df = bias_df[bias_df['model'] == model]

        estimate = model_df['bias'].mean()
        se = model_df['bias'].std() / np.sqrt(len(model_df))
        df = len(model_df) - 1
        ttest = ttest_1samp(model_df['bias'], 0)
        p_value = ttest.pvalue
        bias_test_df.append({'model': model,
                             'estimate': estimate, 'standard_error': se,
                             'test_statistic': ttest.statistic, 'p_value': p_value, 'df': df,})

    bias_test_df = pd.DataFrame(bias_test_df)

    _, p_value_corrected, _, alpha_adjusted = multipletests(bias_test_df['p_value'], alpha=alpha, method='holm')
    bias_test_df['p_value_corrected'] = p_value_corrected
    bias_test_df['significance'] = bias_test_df['p_value_corrected'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    bias_test_df['lower_bound'] = bias_test_df['estimate'] - bias_test_df['standard_error'] * t.ppf(1 - alpha_adjusted / 2, bias_test_df['df'])
    bias_test_df['upper_bound'] = bias_test_df['estimate'] + bias_test_df['standard_error'] * t.ppf(1 - alpha_adjusted / 2, bias_test_df['df'])
    return bias_test_df

def fit_accuracy_bias_mixed_model(df, n_iter=500):
    df = get_subject_level_results(df)
    formula = 'bias ~ confounded_test + (1 | model) + (1 | subject)'
    model = Lmer(formula, data=df)
    model.fit(summary=False, verbose=False)
    regression_df = pd.DataFrame({'confounded_test': np.linspace(df['confounded_test'].min()//5 * 5, (df['confounded_test'].max()//5 + 1) * 5, len(df))})
    regression_df['bias'] = model.predict(regression_df, use_rfx=False)

    confintervals = []
    for _ in tqdm(range(n_iter), desc='Bootstrapping confidence intervals', unit='iteration'):
        sample_df = df.sample(frac=1, replace=True).reset_index(drop=True)
        boot_model = Lmer(formula, data=sample_df)
        boot_model.fit(summarize=False, verbose=False)
        confintervals.append(boot_model.predict(regression_df, use_rfx=False))
    confintervals = np.array(confintervals)
    regression_df['lower_bound'] = np.percentile(confintervals, 2.5, axis=0)
    regression_df['upper_bound'] = np.percentile(confintervals, 97.5, axis=0)

    fixef_df = model.summary()
    ranef_df = model.ranef_var
    confint_df = model.confint(level=0.95, method='boot')

    return fixef_df, ranef_df, confint_df, regression_df

def plot_bias_accuracy_regression(pred_df, regression_df, path=None):
    _, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.grid(True, which='both', axis='both', linestyle='-', linewidth=0.5)

    bias_df = get_subject_level_results(pred_df)

    sns.scatterplot(ax=ax, data=bias_df,
                x='confounded_test', y='bias')

    no_bias = ax.axhline(0, color='black', linestyle='--', linewidth=2, label='No bias')
    chance = ax.axvline(100/6, color='red', linestyle='--', linewidth=2, label='Chance accuracy')
    fit = ax.add_line(plt.Line2D([0], [0], color=sns.color_palette()[0], linestyle='-', linewidth=2, label='LME fit'))
    ax.plot(regression_df['confounded_test'], regression_df['bias'], color=sns.color_palette()[0], linestyle='-', linewidth=2, label='LME fit')
    ci = ax.fill_between(regression_df['confounded_test'], regression_df['lower_bound'], regression_df['upper_bound'], color=sns.color_palette()[0], alpha=0.2, label='95\% CI')

    ref_legend = plt.legend(title='Reference', loc='upper left', fontsize=12, title_fontsize='x-large',
                            handles=[no_bias, chance, fit, ci], labels=['No bias', 'Chance accuracy', 'LME fit', '95\% CI'], # shift to right
                            bbox_to_anchor=(0.05, 0.99))

    # Format x and y axis labels
    ax.set_xlabel('Confounded accuracy (\%)', fontsize=18, labelpad=10)
    ax.set_ylabel('Absolute bias (\%)', fontsize=18, labelpad=10)
    ax.set_xticks(np.arange(20, 65, 5), ['{:d}\%'.format(x) for x in np.arange(20, 65, 5)], fontsize=14)
    ax.set_yticks(np.arange(0, 15, 5), ['{:d}\%'.format(x) for x in np.arange(0, 15, 5)], fontsize=14)
    ax.set_xlim(15, 65)
    ax.set_ylim(-2.5, 15)

    if path:
        plt.savefig(path, format='pdf', bbox_inches='tight')
    else:
        plt.show()

def test_bias_category_dependence(bias_df):
    formula = 'bias ~ category + (1|model) + (1|subject)'
    lme_model = Lmer(data=bias_df, formula=formula)
    lme_model.fit(factors={'category': list(CATEGORY_MAP.values())}, summary=False, verbose=False)
    bias_est_df, bias_contrast_df = lme_model.post_hoc(marginal_vars='category', p_adjust='holm')

    return bias_est_df, bias_contrast_df

def plot_bias_accuracy_scatter_by_category(bias_df, path=None):
    g = sns.FacetGrid(bias_df, col='category', col_wrap=3, aspect = 1.9, col_order = list(CATEGORY_MAP.values()))
    g.map_dataframe(sns.scatterplot, x='unconfounded_test', y='bias', alpha=0.7)

    g.set_xlabels('\Huge Accuracy (\%)', labelpad=10)
    g.set_ylabels('\Huge Bias (\%)', labelpad=10)
    # Add slope and intercept of the regression line to the title
    g.set_titles(col_template="\Huge {col_name}")
    # set xticks
    g.set(xticks=np.arange(20, 81, 20), yticks=np.arange(0, 21, 5))
    g.set_xticklabels(['\Huge {:d}\%'.format(x) for x in np.arange(20, 81, 20)])
    g.set_yticklabels(['\Huge {:d}\%'.format(x) for x in np.arange(0, 21, 5)])

    if path:
        plt.savefig(path, format='pdf', bbox_inches='tight')
    else:
        plt.show()

def plot_bias_accuracy_point_by_category(bias_df, path=None):
    order = bias_df.groupby('category').agg(unconfounded_test=('unconfounded_test', 'mean')).sort_values('unconfounded_test').index

    fig, ax = plt.subplots(2, 1, figsize=(12, 4), constrained_layout=True, sharex=True)
    sns.pointplot(data=bias_df, x='category', y='unconfounded_test', order = order, linewidth=2, ax=ax[0], markersize=10, linestyle='none', errorbar=None, legend=False,
                color=sns.color_palette()[0])
    sns.pointplot(data=bias_df, x='category', y='bias', order = order, linewidth=2, ax=ax[1], markersize=10, linestyle='none', errorbar=None, legend=False,
                color=sns.color_palette()[1])

    ax[0].grid(which='both', axis='y', linestyle='-', linewidth=0.5)
    ax[1].grid(which='both', axis='y', linestyle='-', linewidth=0.5)
    ax[0].set_yticks(np.arange(30, 61, 10), ['{:d}\%'.format(x) for x in np.arange(30, 61, 10)], fontsize=12)

    ax[0].set_ylim(20, 70)
    ax[0].set_ylabel('Avg. accuracy', fontsize=16, labelpad=12)

    ax[1].set_yticks(np.arange(3, 9, 1), ['{:d}\%'.format(x) for x in np.arange(3, 9, 1)], fontsize=12)
    ax[1].set_ylim(2, 9)
    ax[1].set_ylabel('Avg. bias', fontsize=16, labelpad=16)

    ax[1].set_xlabel('Category', fontsize=16, labelpad=8)
    ax[1].set_xticklabels(order, fontsize=14)

    if path:
        plt.savefig(path, format='pdf', bbox_inches='tight')
    else:
        plt.show()

def get_confusion_matrix(df, partition):
    cm = df.groupby(['partition', 'category', 'category_pred']).size().reset_index(name='count')
    cm = cm.pivot_table(index=['partition', 'category'],
                        columns='category_pred',
                        values='count', fill_value=0).reset_index()
    cm = cm[cm['partition'] == partition].set_index('category').drop(columns='partition', axis=1)
    cm = cm.div(cm.sum(axis=1), axis=0).loc[CATEGORY_MAP.values(), CATEGORY_MAP.values()]

    return cm

def plot_confusion_matrices_by_model(df, dir=False):
    for model in MODELS['primary'] + MODELS['additional']:
        model_df = df[df['model'] == model]

        unconfounded_cm = get_confusion_matrix(model_df, 'unconfounded_test')
        confounded_cm = get_confusion_matrix(model_df, 'confounded_test')

        fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True)
        vmin = min(unconfounded_cm.values.min(), confounded_cm.values.min())
        vmax = max(unconfounded_cm.values.max(), confounded_cm.values.max())

        sns.heatmap(confounded_cm, ax=ax[0], annot=confounded_cm.map(lambda x: f"${x:.2f}$"), fmt='', cmap='viridis', cbar=False, vmin=vmin, vmax=vmax)
        sns.heatmap(unconfounded_cm, ax=ax[1], annot=unconfounded_cm.map(lambda x: f"${x:.2f}$"), fmt='', cmap='viridis', cbar=False, vmin=vmin, vmax=vmax)
        sns.heatmap(unconfounded_cm - confounded_cm, ax=ax[2], annot=unconfounded_cm.sub(confounded_cm).map(lambda x: f"${x:.2f}$"), fmt='', cmap='coolwarm', cbar=False)

        for i in range(3):
            ax[i].set_xticklabels(CATEGORY_ABBR_MAP.values(), rotation=45, horizontalalignment='right', fontsize=14)
            ax[i].set_xlabel('Predicted category', fontsize=18, labelpad=10)

        ax[0].set_ylabel('True category', fontsize=18, labelpad=10)
        ax[0].set_yticklabels(CATEGORY_ABBR_MAP.values(), rotation=0, fontsize=14)
        ax[1].set_ylabel('')
        ax[2].set_ylabel('')

        ax[0].set_title('Confounded', fontsize=20)
        ax[1].set_title('Unconfounded', fontsize=20)
        ax[2].set_title('Difference', fontsize=20)

        if dir:
            path = os.path.join(dir, f'{model}_confusion_matrices.pdf')
            plt.savefig(path, format='pdf', bbox_inches='tight')
        else:
            plt.show()

def test_accuracy_significance(df, alpha=0.05):
    accuracy_df = get_subject_level_results(df)
    accuracy_test_df = []
    for model in MODELS['primary'] + MODELS['additional']:
        model_df = accuracy_df[accuracy_df['model'] == model]

        for partition in ['confounded_test', 'unconfounded_test']:
            estimate = model_df[partition].mean()
            se = model_df[partition].std() / np.sqrt(len(model_df))
            ttest = ttest_1samp(model_df[partition], 100/12)
            p_value = ttest.pvalue
            accuracy_test_df.append({'model': model,
                                     'partition': partition,
                                     'estimate': estimate, 'standard_error': se,
                                     'test_statistic': ttest.statistic, 'p_value': p_value})
    accuracy_test_df = pd.DataFrame(accuracy_test_df)
    reject, p_value_corrected, _, alpha_adjusted = multipletests(accuracy_test_df['p_value'], alpha=alpha, method='bonferroni')
    accuracy_test_df['reject_null'] = reject
    accuracy_test_df['p_value_corrected'] = p_value_corrected
    accuracy_test_df['significance'] = accuracy_test_df['p_value_corrected'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')
    accuracy_test_df['lower_bound'] = accuracy_test_df['estimate'] - accuracy_test_df['standard_error'] * t.ppf(1 - alpha_adjusted / 2, len(model_df) - 1)
    accuracy_test_df['upper_bound'] = accuracy_test_df['estimate'] + accuracy_test_df['standard_error'] * t.ppf(1 - alpha_adjusted / 2, len(model_df) - 1)

    accuracy_test_df = accuracy_test_df.pivot(index='model', columns='partition', values=['estimate', 'standard_error', 'reject_null', 'test_statistic', 'p_value', 'p_value_corrected', 'significance', 'lower_bound', 'upper_bound'])

    return accuracy_test_df

def analyze_results(paired_category_decoding, paired_pseudocategory_decoding):

    data_dir = os.path.join('outputs', 'paired_category_decoding')
    fig_dir = 'figures'
    os.makedirs(fig_dir, exist_ok=True)

    pred_df = get_predictions(paired_category_decoding)
    bias_df = get_model_level_results(pred_df)
    bias_df.to_csv(os.path.join(data_dir, 'category_decoding_bias_summary.csv'), index=False)

    plot_bias_boxplot_by_model_and_subject(pred_df, model_type='primary', path=os.path.join(fig_dir, 'category_decoding_bias_boxplot_by_primary_model_and_subject.pdf'))
    bias_test_df = test_bias_significance(pred_df, alpha=0.05)
    bias_test_df.to_csv(os.path.join(data_dir, 'category_decoding_bias_significance.csv'), index=False)

    fixef_df, ranef_df, confint_df, regression_df = fit_accuracy_bias_mixed_model(pred_df, n_iter=5)
    fixef_df.to_csv(os.path.join(data_dir, 'bias_accuracy_mixed_model_fixed_effects.csv'))
    ranef_df.to_csv(os.path.join(data_dir, 'bias_accuracy_mixed_model_random_effects.csv'))
    confint_df.to_csv(os.path.join(data_dir, 'bias_accuracy_mixed_model_confidence_intervals.csv'))

    plot_bias_accuracy_regression(pred_df, regression_df, path=os.path.join(fig_dir, 'bias_accuracy_mixed_model_regplot.pdf'))

    category_bias_df = get_subject_level_category_results(pred_df)

    bias_category_est_df, bias_category_contrast_df = test_bias_category_dependence(category_bias_df)
    bias_category_est_df.to_csv(os.path.join(data_dir, 'category_bias_estimates.csv'), index=False)
    bias_category_contrast_df.to_csv(os.path.join(data_dir, 'category_bias_contrasts.csv'), index=False)

    plot_bias_accuracy_scatter_by_category(category_bias_df, path=os.path.join(fig_dir, 'bias_accuracy_scatter_by_category.pdf'))

    plot_bias_accuracy_point_by_category(category_bias_df, path=os.path.join(fig_dir, 'bias_accuracy_point_by_category.pdf'))
    cm_dir = os.path.join(fig_dir, 'confusion_matrices')
    if not os.path.exists(cm_dir):
        os.makedirs(cm_dir)
    plot_confusion_matrices_by_model(pred_df, dir=cm_dir)

    data_dir = os.path.join('outputs', 'paired_pseudocategory_decoding')
    pred_df = get_predictions(paired_pseudocategory_decoding)
    bias_df = get_model_level_results(pred_df)
    bias_df.to_csv(os.path.join(data_dir, 'pseudocategory_decoding_bias_summary.csv'), index=False)

    acc_test_df = test_accuracy_significance(pred_df, alpha=0.05)
    acc_test_df.to_csv(os.path.join(data_dir, 'pseudocategory_decoding_accuracy_significance.csv'))
