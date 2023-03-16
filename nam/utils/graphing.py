from typing import Sequence
from typing import Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

features_names = ['b', 'h', 'd', "d'", 'L', 'l', 'a', "fck", "As", "fy", "As'", "fy'",
        'C', 'PBO', 'G', 'CF', 'B', 'ff', 'Af', 'layer', 'Swr', 'Swf',
        'Anchorage']

def get_feature_contributions(model, x_test,device):
    features = x_test.float().to(device)
    single_features = np.split(np.array(features), features.shape[1], axis=1)
    unique_features = [np.unique(f, axis=0) for f in single_features]
    feature_contributions = []
    for i, feature in enumerate(unique_features):
        feature = torch.tensor(feature).float().to(device)
        feat_contribution = model.feature_nns[i](feature).cpu().detach().numpy().squeeze()
        feature_contributions.append(feat_contribution)
    
    return feature_contributions, unique_features, single_features


def calc_mean_prediction(model, x_test, device):


    feature_contributions, unique_features,_ = get_feature_contributions(model, x_test,device)
    avg_hist_data = {col: contributions for col, contributions in zip(features_names, feature_contributions)}
    all_indices, mean_pred = {}, {}

    features = x_test.float().to(device)

    for i, col in enumerate(features_names):
        feature_i = features[:, i].cpu()
        all_indices[col] = np.searchsorted(unique_features[i][:, 0], feature_i, 'left')

    for col in features_names:
        mean_pred[col] = np.mean([avg_hist_data[col]])  #[i] for i in all_indices[col]]) TODO: check the error here

    return mean_pred, avg_hist_data


def plot_mean_feature_importance(model, x_test, device, width=0.5):
    mean_pred, avg_hist_data = calc_mean_prediction(model, x_test, device)

    def compute_mean_feature_importance(mean_pred, avg_hist_data):
        mean_abs_score = {}
        for k in avg_hist_data:
            try:
                mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - mean_pred[k]))
            except:
                continue
        x1, x2 = zip(*mean_abs_score.items())
        return x1, x2

    ## TODO: rename x1 and x2
    x1, x2 = compute_mean_feature_importance(mean_pred, avg_hist_data)

    cols = features_names
    fig = plt.figure(figsize=(12, 5))
    ind = np.arange(len(x1))
    x1_indices = np.argsort(x2)
    
    cols_here = [cols[i] for i in x1_indices]
    x2_here = [x2[i] for i in x1_indices]
    
    plt.bar(ind, x2_here, width, label='NAMs')
    plt.xticks(ind + width / 2, cols_here, rotation=90, fontsize='large')
    plt.ylabel('Mean Absolute Score', fontsize='x-large')
    plt.legend(loc='upper right', fontsize='large')
    plt.title(f'Overall Importance', fontsize='x-large')
    plt.show()
    
    return fig

def plot_nams(model,
              x_test,
              num_cols,
              n_blocks,
              color,
              linewidth,
              alpha,
              feature_to_use,
              mean_dict,
              std_dict,
              device):
    
    features = x_test.float().to(device)
    _, unique_features_norm, single_features = get_feature_contributions(model, x_test, device)
    single_features = np.split(np.array(features), features.shape[1], axis=1)
    unique_features_norm = [np.unique(f, axis=0) for f in single_features]
    unique_features = {}
    for idx, feature_name in enumerate(features_names):
        unique_features[feature_name] = (unique_features_norm[idx]*std_dict[feature_name])+mean_dict[feature_name]
    mean_pred, feat_data_contrib = calc_mean_prediction(model, x_test,device)

    num_rows = len(features[0]) // num_cols

    fig = plt.figure(num=None, figsize=(num_cols * 10, num_rows * 10), facecolor='w', edgecolor='k')
    fig.tight_layout(pad=7.0)

    feat_data_contrib_pairs = list(feat_data_contrib.items())
    feat_data_contrib_pairs.sort(key=lambda x: x[0])

    mean_pred_pairs = list(mean_pred.items())
    mean_pred_pairs.sort(key=lambda x: x[0])

    if feature_to_use:
        feat_data_contrib_pairs = [v for v in feat_data_contrib_pairs if v[0] in feature_to_use]

    min_y = np.min([np.min(a[1]) for a in feat_data_contrib_pairs])
    max_y = np.max([np.max(a[1]) for a in feat_data_contrib_pairs])

    min_max_dif = max_y - min_y
    min_y = min_y - 0.5 * min_max_dif
    max_y = max_y + 0.5 * min_max_dif

    total_mean_bias = 0
    temp_unique = {}
    for name, item in zip(unique_features.keys(), unique_features.values()):
        temp_unique[name] = item

    for i, (name, feat_contrib) in enumerate(feat_data_contrib_pairs):
        mean_pred = mean_pred_pairs[i][1]
        total_mean_bias += mean_pred

        unique_feat_data = temp_unique[name]
        ax = plt.subplot(num_rows, num_cols+1, i + 1)
        plt.plot(unique_feat_data, feat_contrib - mean_pred, color=color, linewidth=linewidth, alpha=alpha)

        plt.xticks(fontsize='x-large')

        plt.ylim(min_y, max_y)
        plt.yticks(fontsize='x-large')

        min_x = np.min(unique_feat_data)  # - 0.5  ## for categorical
        max_x = np.max(unique_feat_data)  # + 0.5
        plt.xlim(min_x, max_x)

        if i % num_cols == 0:
            plt.ylabel('Features Contribution', fontsize='x-large')

        plt.xlabel(name, fontsize='x-large')

    plt.show()
    
    return fig
