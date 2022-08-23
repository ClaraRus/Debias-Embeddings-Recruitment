import numpy as np
import tqdm
import pandas as pd
import torch


def compute_fairness_metrics_batch_multiple_class(preds, data_set, class_names, batch_size):
    X, Y, gender = data_set
    val_index = np.arange(len(Y))
    val_batches = [(i * batch_size, min(X.shape[0], (i + 1) * batch_size))
                   for i in range((X.shape[0] + batch_size - 1) // batch_size)]

    preds = preds.round()

    parities = []
    odds = []
    opps = []
    for iter, (batch_start, batch_end) in tqdm.tqdm(enumerate(val_batches)):
        batch_ids = val_index[batch_start:batch_end]
        x_batch = preds[batch_start:batch_end]
        y_batch = Y[batch_start:batch_end]
        s_batch = gender[batch_start:batch_end]

        data = [x_batch.detach().numpy(), y_batch.detach().numpy(), s_batch.detach().numpy()]
        stat_parity = compute_statistical_parity_multiple_classes(data)
        eql_opp = equal_opportunity_multiple_classes(data, n_classes=len(class_names), positive_target=1)
        equal_odds = equal_odds_multiple_classes(data, n_classes=len(class_names))

        odds.extend(equal_odds)
        opps.extend(eql_opp)
        parities.extend(stat_parity)

    stat_parity = np.mean(parities, axis=0)
    eql_opp = np.mean(opps, axis=0)
    equal_odds = np.mean(odds, axis=0)

    result_df = pd.DataFrame(columns=['Class', 'Statistical Parity', 'Equality Opportunity', 'Equality Odds'])
    result_df['Class'] = class_names
    result_df['Statistical Parity'] = stat_parity
    result_df['Equality Opportunity'] = eql_opp
    result_df['Equality Odds'] = equal_odds

    return result_df


def compute_statistical_parity_multiple_classes(data_set):
    X, Y, gender = data_set
    gender = gender[:, 0]
    male_preds = X[gender == 1]
    female_preds = X[gender == 0]

    male_result = sum(male_preds.round()) / sum(gender == 1)
    female_result = sum(female_preds.round()) / sum(gender == 0)

    disc = abs(male_result - female_result)
    return disc


def equal_opportunity_multiple_classes(data_set, n_classes, positive_target=1):
    X, Y, gender = data_set
    gender = gender[:, 0]

    if not np.all((gender == 0) | (gender == 1)):
        raise ValueError(
            f"equal_opportunity_score only supports binary indicator columns for `column`. "
            f"Found values {np.unique(gender)}"
        )

    scores_classes = []
    for i in range(n_classes):
        mask_z1_y1 = np.logical_and(gender == 1, Y[:, i] == positive_target)
        mask_z0_y1 = np.logical_and(gender == 0, Y[:, i] == positive_target)

        y_hat = X.round()

        y_given_z1_y1 = y_hat[mask_z1_y1]
        y_given_z0_y1 = y_hat[mask_z0_y1]

        # If we never predict a positive target for one of the subgroups, the model is by definition not
        # fair so we return 0
        if len(y_given_z1_y1) == 0:
            score = 0
        elif len(y_given_z0_y1) == 0:
            score = 0
        else:
            p_y1_z1 = np.mean(y_given_z1_y1[:, i] == positive_target)
            p_y1_z0 = np.mean(y_given_z0_y1[:, i] == positive_target)

            if p_y1_z1 == 0 or p_y1_z0 == 0:
                score = 0
            else:
                score = np.minimum(p_y1_z1 / p_y1_z0, p_y1_z0 / p_y1_z1)

        scores_classes.append(score if not np.isnan(score) else 1)

    return scores_classes


def equal_odds_multiple_classes(data_set, n_classes):
    X, Y, gender = data_set
    gender = gender[:, 0]

    scores_classes = []
    for i in range(n_classes):
        mask_z1 = gender == 1
        mask_z0 = gender == 0

        y_hat = X.round()

        y_given_z1 = y_hat[mask_z1]
        y_given_z0 = y_hat[mask_z0]

        if len(y_given_z1) == 0:
            score = 0
        elif len(y_given_z0) == 0:
            score = 0
        else:
            p_z1 = np.mean(y_given_z1[:, i] == Y[mask_z1])
            p_z0 = np.mean(y_given_z0[:, i] == Y[mask_z0])

            if p_z1 == 0 or p_z0 == 0:
                score = 0
            else:
                score = np.minimum(p_z1 / p_z0, p_z0 / p_z1)

        scores_classes.append(score if not np.isnan(score) else 1)

    return scores_classes