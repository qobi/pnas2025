import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import rcParams
import seaborn as sns
from src.data.cross_validation.utils import load_folds
rcParams["text.usetex"] = True
subject = "S1"
fold_idx = 0
category = torch.load(f"data/processed/{subject}/category.pt", weights_only=False)
exemplar = torch.load(f"data/processed/{subject}/exemplar.pt", weights_only=False)

group_group_params = {'target': 'category',
                      'subject_scheme': 'subject_dependent',
                      'subject': subject,
                      'hyperparameter_tuning_scheme': 'grouped',
                      'cross_validation_scheme': 'grouped',
                      'constraint': 'exemplar_balanced',
                      'retrain': True}
group_group_folds = load_folds(subject, group_group_params)

strat_strat_params = {'target': 'category',
                      'subject_scheme': 'subject_dependent',
                      'subject': subject,
                      'hyperparameter_tuning_scheme': 'stratified',
                      'cross_validation_scheme': 'stratified',
                      'constraint': 'exemplar_balanced',
                      'retrain': True}
strat_strat_folds = load_folds(subject, strat_strat_params)
repeated_trials = torch.cat([strat_strat_folds[0]['test'][torch.isin(strat_strat_folds[0]['test'], group_group_folds[0]['val'][0])],
                                strat_strat_folds[0]['test'][torch.isin(strat_strat_folds[0]['test'], group_group_folds[0]['train'][0])]])

print(f"Number of repeated trials: {len(repeated_trials)}")
# Sort first by exemplar value, then by appearance in strat_strat nested val partitions
trial_idxs = torch.cat([group_group_folds[0]['train'][0], group_group_folds[0]['val'][0], group_group_folds[0]['test'], strat_strat_folds[0]['test']])

stratified_nested_val_index = {idx.item(): nested_fold_idx for nested_fold_idx, nested_fold in enumerate(strat_strat_folds[0]['val']) for idx in nested_fold}
grouped_nested_val_index = {idx.item(): nested_fold_idx for nested_fold_idx, nested_fold in enumerate(group_group_folds[0]['val']) for idx in nested_fold}

sorting_df = pd.DataFrame({'trial_idx': trial_idxs, 
                           'exemplar': exemplar[trial_idxs], 
                           'group_val_order': trial_idxs.clone().apply_(lambda idx: grouped_nested_val_index.get(idx, -1)),
                           'strat_val_order': trial_idxs.clone().apply_(lambda idx: stratified_nested_val_index.get(idx, -1))})
sorting_df = sorting_df.sort_values(by=['exemplar', 'strat_val_order', 'group_val_order'])
sorting_df['Order'] = range(len(sorting_df['trial_idx']))
trial_position_map = {trial_idx: order for trial_idx, order in zip(sorting_df['trial_idx'], sorting_df['Order'])}

# sorted_trials = trial_idxs[torch.argsort(exemplar[trial_idxs], descending=False)]
# trial_position_map = {trial_idx.item(): sorted_idx for trial_idx, sorted_idx in zip(sorted_trials, range(len(trial_idxs)))}

exemplar_values, exemplar_counts = exemplar[trial_idxs].unique(return_counts=True)
n_exemplars = len(exemplar_values)

exemplar_bins = np.concatenate([[0], exemplar_counts.numpy()]).cumsum()
min_exemplar_position = exemplar_bins[:-1]
max_exemplar_position = exemplar_bins[1:]

exemplar_color = np.array(sns.color_palette("husl", n_exemplars))

def format_partition(cv_data, scheme_params, fold_idx, nested_fold_idx, partition, category, exemplar, trial_idx):
    for key, val in scheme_params.items():
        cv_data[key].extend([val] * len(trial_idx))
    cv_data["Fold Index"].extend([fold_idx] * len(trial_idx))
    cv_data["Nested Fold Index"].extend([nested_fold_idx] * len(trial_idx))
    cv_data["Partition"].extend([partition] * len(trial_idx))
    cv_data["Category"].extend(category[trial_idx].tolist())
    cv_data["Exemplar"].extend(exemplar[trial_idx].tolist())
    cv_data["Trial Index"].extend(trial_idx.tolist())
    cv_data['Fold : Partition'].extend([f"Fold {fold_idx} : {partition}"] * len(trial_idx))
    cv_data['Nested Fold : Partition'].extend([partition if nested_fold_idx is None else f"Nested Fold {nested_fold_idx} : {partition}"] * len(trial_idx))
    return cv_data

def build_cv_df(folds, scheme_params, category, exemplar):
    scheme_params = {' '.join([w.capitalize() for w in key.split('_')]): val.capitalize() if isinstance(val, str) else val for key, val in scheme_params.items()}
    cv_data = {key: [] for key in scheme_params.keys()}
    cv_data.update({"Fold Index": [], "Nested Fold Index": [], "Partition": [], "Category": [], "Exemplar": [], "Trial Index": []})
    cv_data.update({'Fold : Partition': [], 'Nested Fold : Partition': []})
    for fold_idx, fold in enumerate(folds):
        test_idx = fold['test']
        cv_data = format_partition(cv_data, scheme_params, fold_idx, None, "Test", category, exemplar, test_idx)
        for nested_fold_idx, (train_idx, val_idx) in enumerate(zip(fold['train'], fold['val'])):
            cv_data = format_partition(cv_data, scheme_params, fold_idx, nested_fold_idx, "Train", category, exemplar, train_idx)
            cv_data = format_partition(cv_data, scheme_params, fold_idx, nested_fold_idx, "Validation", category, exemplar, val_idx)
    cv_df = pd.DataFrame(cv_data)
    cv_df = set_cv_df_column_properties(cv_df)
    return cv_df

def set_cv_df_column_properties(cv_df):
    cv_df["Partition"] = pd.Categorical(cv_df["Partition"], categories=["Train", "Validation", "Test"], ordered=True)
    cv_df["Category"] = pd.Categorical(cv_df["Category"], categories=cv_df["Category"].unique().sort(), ordered=True)
    cv_df["Exemplar"] = pd.Categorical(cv_df["Exemplar"], categories=cv_df["Exemplar"].unique().sort(), ordered=True)
    return cv_df

strat_strat_df = build_cv_df(strat_strat_folds, strat_strat_params, category, exemplar)
group_group_df = build_cv_df(group_group_folds, group_group_params, category, exemplar)

cv_df = pd.concat([strat_strat_df, group_group_df])
cv_df["Position"] = cv_df["Trial Index"].map(trial_position_map)

fold_df = cv_df[cv_df["Fold Index"] == fold_idx]
# fold_df = fold_df[fold_df["Hyperparameter Tuning Scheme"] == "Grouped"]

def style_y_axis(ax, fold_df):
    # Grouped Test
    # Stratified Test 
    # ...
    # Stratified Validation - 8
    # Stratified Train      - 7
    #                       - 6
    # Nested Fold 0         - 5
    # Grouped Validation    - 4
    # Grouped Train         - 3
    #                       - 2
    # Stratified Validation - 1
    # Stratified Train      - 0
    schemes = ['Stratified', 'Grouped']
    # nested_folds = fold_df['Nested Fold Index'][fold_df['Nested Fold Index'].notnull()].unique()
    nested_folds = ["0"]
    partitions = ["Validation", "Train"]
    
    partition_y_locs = {}
    nested_fold_y_locs = {}
    loc = 1

    for fdx in nested_folds:
        for scheme in schemes:
            for partition in partitions:
                partition_y_locs[f"{fdx}-{scheme}-{partition}"] = loc
                loc += 2
            nested_fold_y_locs[f"{scheme}-{fdx}"] = loc - 1
            loc += 1
        nested_fold_y_locs[fdx] = loc
        loc+= 2
        
    nested_fold_y_locs[f"Stratified-Test"] = loc + 1
    partition_y_locs[f"Stratified-Test"] = loc + 1.01

    nested_fold_y_locs[f"Grouped-Test"] = loc + 4
    partition_y_locs[f"Grouped-Test"] = loc + 4.01

    ax.set_yticks(list(nested_fold_y_locs.values()))
    ax.set_yticklabels([r"{{\Large \textbf{{Fold {0}}}}}\hspace{{2em}}".format(int(k)) if isinstance(k, float) else r"{{\large \begin{{itemize}}\item {0}\end{{itemize}}}}".format(k.split('-')[0].capitalize()) for k in nested_fold_y_locs.keys()], ha='left')

    ax.set_yticks(list(partition_y_locs.values()), minor=True)
    y_tick_labels = [r"{0}".format(k.split('-')[-1].capitalize()) for k in partition_y_locs.keys()]
    y_tick_labels[-1] = r"{{\Large Test}}"
    y_tick_labels[-2] = r"{{\Large Test}}"
    ax.set_yticklabels(y_tick_labels, minor=True)
    ax.set_ylim(-1, max(partition_y_locs.values()) + 2)
    
    ax.set_xlim(0, fold_df["Position"].max())

    ax.tick_params(axis='y', which='major', length=0, width=1.5, labelsize=16, right=False, left=True, labelleft=True, labelright=False, pad=64)
    ax.tick_params(axis='y', which='minor', length=3, width=1.5, labelsize=14, right=True, left=False, labelleft=False, labelright=True)

    ax.grid(axis='y', which='major', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(axis='y', which='minor', linestyle='--', linewidth=0.5, alpha=0.5)

    return partition_y_locs

def set_xtickimages(ax, xticks, images, edgecolors, offset=0):
    
    for i, (x, img) in enumerate(zip(xticks, images)):
        imagebox = OffsetImage(img, zoom=0.125)
        ab = AnnotationBbox(imagebox, (x, -offset),
                            frameon=True,
                            box_alignment=(0.5, 0.5),
                            annotation_clip=False,
                            pad=3,
                            bboxprops=dict(edgecolor=edgecolors[i], linewidth=3, boxstyle="round,pad=0.0825"))
        ax.add_artist(ab)

def style_x_axis(ax, fold_df, exemplar_color):
    
    exemplar_x_locs = fold_df.groupby("Exemplar", observed=False)["Position"].median().sort_values().values
    exemplar_images = [plt.imread(f"figures/stimuli/reformatted/stimulus{i+1}.png") for i in range(72)]

    ax.set_xticks(exemplar_x_locs[1::2])
    ax.set_xticklabels([])
    set_xtickimages(ax, exemplar_x_locs[1::2], exemplar_images[1::2], exemplar_color[1::2], offset=5)
    ax.tick_params(axis='x', which='major', length=40, width=1.5)

    ax.set_xticks(exemplar_x_locs[::2], minor=True)
    ax.set_xticklabels([])
    set_xtickimages(ax, exemplar_x_locs[::2], exemplar_images[::2], exemplar_color[::2], offset=3.5)
    ax.tick_params(axis='x', which='minor', length=20, width=1.5)

    for idx, tick in enumerate(ax.get_xticklines()):
        tick.set_color(exemplar_color[idx])
        tick.set_markeredgecolor(exemplar_color[idx])

    for idx, tick in enumerate(ax.get_xticklines(minor=True)):
        tick.set_color(exemplar_color[idx])
        tick.set_markeredgecolor(exemplar_color[idx])

    grid_x_locs = fold_df.groupby("Exemplar", observed=False)["Position"].max().sort_values().values
    for x in grid_x_locs:
        ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')

def plot_all_schemes_by_partition(fold_df, save=False):

    _, ax = plt.subplots(figsize=(20, 20))

    strat_gradients = [
        ((0.1, 0.4, 0.8), (0.2, 0.6, 0.9)),  # Blue gradient
        ((0.9, 0.5, 0.2), (1.0, 0.8, 0.5)),  # Orange gradient
        ((0.9, 0.1, 0.2), (0.8, 0.3, 0.4)),  # Red gradient
        ((0.2, 0.8, 0.3), (0.4, 1.0, 0.6)),  # Green gradient
        ((0.5, 0.1, 0.7), (0.6, 0.3, 0.9)),  # Purple gradient
        ((0.9, 0.8, 0.2), (1.0, 1.0, 0.6)),  # Yellow gradient
    ]

    group_gradients = [
        ((0.3, 0.2, 0.8), (0.5, 0.6, 1.0)),  # Violet gradient
        ((0.8, 0.4, 0.2), (1.0, 0.6, 0.4)),  # Copper gradient
        ((0.8, 0.2, 0.3), (1.0, 0.5, 0.6)),  # Rose gradient
        ((0.3, 0.7, 0.4), (0.5, 1.0, 0.7)),  # Aqua gradient
        ((0.6, 0.3, 0.7), (0.8, 0.5, 1.0)),  # Lilac gradient
        ((1.0, 0.9, 0.4), (1.0, 1.0, 0.7)),  # Gold gradient
    ]

    mixed_gradients = []
    for ((ssr, ssg, ssb), (ser, seg, seb)), ((gsr, gsg, gsb), (ger, geg, geb)) in zip(strat_gradients, group_gradients):
        mixed_gradients.append((((ssr + gsr) / 2, (ssg + gsg) / 2, (ssb + gsb) / 2), ((ser + ger) / 2, (seg + geg) / 2, (seb + geb) / 2)))

    
    strat_colors =  np.concatenate([np.array(sns.blend_palette(sg, n_colors=12)) for sg in strat_gradients])
    group_colors = np.concatenate([np.array(sns.blend_palette(gg, n_colors=12)) for gg in group_gradients])
    mixed_colors = np.concatenate([np.array(sns.blend_palette(mg, n_colors=12)) for mg in mixed_gradients])
    
    fold_df['color'] = fold_df.apply(lambda row: strat_colors[row["Exemplar"]] if row["Hyperparameter Tuning Scheme"] == "Stratified" else group_colors[row["Exemplar"]], axis=1)
    partition_y_loc_map = style_y_axis(ax, fold_df)
    fold_df['y'] = fold_df.apply(lambda row: f"{row['Hyperparameter Tuning Scheme']}-{row['Partition']}" if row["Partition"] == "Test" else f"{row['Nested Fold Index']}-{row['Cross Validation Scheme']}-{row['Partition']}", axis=1)

    fold_df['y_loc'] = fold_df['y'].map(partition_y_loc_map)

    style_x_axis(ax, fold_df, mixed_colors)

    ax.barh(y=fold_df["y_loc"],
            width=1,
            left=fold_df["Position"],
            height=1,
            color=fold_df["color"],
            linewidth=0)

    plt.tight_layout()

    plt.savefig("figures/cross_validation/cv_viz.pdf", format="pdf")

def plot_nested_fold_trials_by_exemplars(nested_fold_df):

    gradients = [
        ((0.1, 0.4, 0.8), (0.2, 0.6, 0.9)),  # Blue gradient
        ((0.9, 0.5, 0.2), (1.0, 0.8, 0.5)),  # Orange gradient
        ((0.9, 0.1, 0.2), (0.8, 0.3, 0.4)),  # Red gradient
        ((0.2, 0.8, 0.3), (0.4, 1.0, 0.6)),  # Green gradient
        ((0.5, 0.1, 0.7), (0.6, 0.3, 0.9)),  # Purple gradient
        ((0.9, 0.8, 0.2), (1.0, 1.0, 0.6)),  # Yellow gradient
    ]
    exemplar_colors =  np.concatenate([np.array(sns.blend_palette(grad, n_colors=12)) for grad in gradients])
    _, ax = plt.subplots(3, 1, figsize=(20, 10))
    for pdx, partition in enumerate(nested_fold_df["Partition"].unique()):
        partition_df = nested_fold_df[nested_fold_df["Partition"] == partition]
        loc_df = partition_df.sort_values(by=["Hyperparameter Tuning Scheme", "Cross Validation Scheme", "Partition", "Exemplar"]).reset_index(drop=True)
        x_locs = loc_df.groupby(["Hyperparameter Tuning Scheme", "Cross Validation Scheme"]).transform('cumcount')

        scheme_index = {f'{hp}-{cv}': i + 2*j for i, hp in enumerate(['Stratified', 'Grouped']) for j, cv in enumerate(['Stratified', 'Grouped'])}
        y_locs = loc_df.apply(lambda row: scheme_index[f'{row['Hyperparameter Tuning Scheme']}-{row['Cross Validation Scheme']}'], axis = 1)

        ax[pdx].barh(y=y_locs,
                width=1,
                left=x_locs,
                height=1,
                color=exemplar_colors[loc_df["Exemplar"]],
                linewidth=0)
        ax[pdx].set_xlabel("Trials (ordered by stimulus)", labelpad=22, fontsize=16)
        ax[pdx].set_ylabel("Scheme", fontsize=16)

    # ax.set_yticks(scheme_index.values())
    # ax.set_yticklabels(scheme_index.keys())
    plt.show()

nested_fold_df = fold_df[(fold_df["Nested Fold Index"] == 0) | (fold_df["Partition"] == "Test")]
# plot_nested_fold_trials_by_exemplars(fold_df)
plot_all_schemes_by_partition(fold_df)

# def plot_fold_exemplars_by_partition(scheme_params, fold_df, save=False):

#     _, ax = plt.subplots(figsize=(20, 10))
#     bar_locs = {f"Nested Fold {int(f)} : {p}": (f)*4 + j - 0.5 for j, p in enumerate(["Validation", "Train"]) for f in range(len(fold_df['Nested Fold Index'].unique()) - 1)}
#     bar_locs["Test"] = max(bar_locs.values()) + 3

#     ax.barh(y=fold_df["Nested Fold : Partition"].map(bar_locs),
#             width=1,
#             left=fold_df["Position"],
#             height=1,
#             color=exemplar_color[fold_df["Exemplar"].to_numpy()],
#             linewidth=0)
    
#     style_plot(fold_df, ax, bar_locs, exemplar_color)

#     if save:
#         file_name = f"{' '.join(w.capitalize() for w in scheme_params['task'].split('_'))} - with {scheme_params['hyperparameter_tuning_scheme']} Hyperparameter Tuning and {scheme_params['cross_validation_scheme']} Cross-validation for Subject {subject[1:]}, Fold {fold_idx}"
#         path = f"figures/cross_validation/{file_name}.pdf"
#         plt.savefig(path, format="pdf")
#     else:
#         plt.show()

# def set_images_as_ticklabel(ax, xticks, images, edgecolors, offset=0):
#     for i, (x, img) in enumerate(zip(xticks, images)):
#         imagebox = OffsetImage(img, zoom=0.125)
#         ab = AnnotationBbox(imagebox, (x, -offset),
#                             frameon=True,  # No box around the image
#                             box_alignment=(0.5, 0.5),
#                             annotation_clip=False,
#                             pad=3,
#                             bboxprops=dict(edgecolor=edgecolors[i], linewidth=3, boxstyle="round,pad=0.0825"))
#         ax.add_artist(ab)

# def style_plot(fold_df, ax, bar_locs, exemplar_color):
#     ax.set_xlabel("Trials (ordered by stimulus)", labelpad=22, fontsize=16)
#     ax.set_ylabel("Nested Fold", fontsize=16)

#     for axis in ['top','bottom','left','right']:
#         ax.spines[axis].set_linewidth(1.5)
    
#     major_y_ticks = [2 + 4*(i) for i in range(11)]
#     major_y_ticklabels = [f"Fold {i}" for i in range(11)]
#     ax.set_yticks(major_y_ticks)
#     ax.set_yticklabels(major_y_ticklabels)

#     ax.tick_params(axis='y', which='major', 
#                 length=5, width=1.5, 
#                 labelsize=12, 
#                 right=False, left=True, labelleft=True, labelright=False)

#     minor_y_ticks = np.array(list(bar_locs.values()))
#     minor_y_ticklabels = np.concatenate([np.array(["Validation", "Train"]).repeat(11), np.array(["Test"])])
#     ax.set_yticks(minor_y_ticks, minor=True)
#     ax.set_yticklabels(minor_y_ticklabels, minor=True)

#     ax.tick_params(axis='y', which='minor', 
#                    length=3, width=1.5, 
#                    labelsize=12, 
#                    right=True, left=False, labelright=True, labelleft=False)



#     major_x_ticks = np.vstack([min_exemplar_position[::2], max_exemplar_position[::2]]).mean(axis=0)
#     ax.set_xticks(major_x_ticks)
#     ax.set_xticklabels([])
#     ax.tick_params(axis='x', which='major', length=20, width=1.5)

#     for idx, tick in enumerate(ax.get_xticklines()):
#         tick.set_color(exemplar_color[idx])
#         tick.set_markeredgecolor(exemplar_color[idx])

#     minor_x_ticks = np.vstack([min_exemplar_position[1::2], max_exemplar_position[1::2]]).mean(axis=0)
#     ax.set_xticks(minor_x_ticks, minor=True)
#     ax.set_xticklabels([], minor=True)
#     ax.tick_params(axis='x', which='minor', length=40, width=1.5)

#     for idx, tick in enumerate(ax.get_xticklines(minor=True)):
#         tick.set_color(exemplar_color[idx])
#         tick.set_markeredgecolor(exemplar_color[idx])

#     for x in min_exemplar_position:
#         ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')
    
#     for y in [i for i in range(-1, 46) if i % 4 != 2]:
#         ax.axhline(y, color='gray', linewidth=0.1, alpha=0.25, linestyle='-') 

#     ax.set_xlim(0, max_exemplar_position.max())
#     ax.set_ylim(-2, max(bar_locs.values()) + 1)
    
#     ax.grid(axis='y', which='major', linestyle='-', linewidth=0.5, alpha=0.5)


#     images = [plt.imread(f"figures/stimuli/reformatted/stimulus{i+1}.png") for i in range(72)]
    
#     set_images_as_ticklabel(ax, major_x_ticks, 
#                             images[::2], offset=3.5, 
#                             edgecolors=exemplar_color[::2])
    
#     set_images_as_ticklabel(ax,
#                             minor_x_ticks, 
#                             images[1::2], offset=6, 
#                             edgecolors=exemplar_color[1::2])
    
#     title = f"{' '.join(w.capitalize() for w in scheme_params['task'].split('_'))} Trials by Nested Fold"
#     subtitle = f"{scheme_params['hyperparameter_tuning_scheme'].capitalize()} Hyperparameter Tuning and {scheme_params['cross_validation_scheme'].capitalize()} Cross-validation"
#     plt.title(subtitle, y=1, fontsize=18)
#     plt.suptitle(title, y = 0.95, fontsize=24)


# for hyperparameter_tuning_scheme in ['stratified', 'grouped']:
#     for cross_validation_scheme in ['stratified', 'grouped']:

#         scheme_params = {'task': 'category_decoding',
#                         'subject_scheme': 'subject_dependent',
#                         'subject': subject,
#                         'hyperparameter_tuning_scheme': hyperparameter_tuning_scheme,
#                         'cross_validation_scheme': cross_validation_scheme,
#                         'constraint': 'exemplar_balanced',
#                         'retrain': True}
#         scheme_folds = load_folds(scheme_params)

#         cv_df = build_cv_df(scheme_folds, scheme_params, category, exemplar)
#         cv_df["Position"] = cv_df["Trial Index"].map(trial_position_map)
        
#         fold_df = cv_df[cv_df["Fold Index"] == fold_idx]

#         plot_fold_exemplars_by_partition(scheme_params, fold_df, save=True)