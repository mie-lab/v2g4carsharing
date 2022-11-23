import pandas as pd


def prepare_data(trips, min_number_trips=200, return_normed=True, drop_columns=[]):
    # drop geometry if it exists
    dataset = trips.drop(["geom", "geom_origin", "geom_destination"] + drop_columns, axis=1, errors="ignore")
    print("Dataset raw", len(dataset))
    # only include frequently used modes (now it usually uses all because we preselect modes earlier!)
    nr_trips_with_mode = trips[[col for col in trips.columns if col.startswith("Mode")]].sum()
    included_modes = list(nr_trips_with_mode[nr_trips_with_mode > min_number_trips].index.tolist())
    print("included_modes", included_modes)
    # TODO: group into public transport, slow transport, car, shared car

    # only get feature and label columns
    feat_cols = [col for col in dataset.columns if col.startswith("feat")]
    dataset = dataset[feat_cols + included_modes]

    # drop columns with too many nans:
    max_unavail = 0.1  # if more than 10% are NaN
    feature_avail_ratio = pd.isna(dataset).sum() / len(dataset)
    features_not_avail = feature_avail_ratio[feature_avail_ratio > max_unavail].index
    dataset.drop(features_not_avail, axis=1, inplace=True)
    print("dataset len now", len(dataset))

    # remove other NaNs (the ones because of missing origin or destination ID)
    dataset.dropna(inplace=True)
    print("dataset len after dropna", len(dataset))

    # convert features to array
    feat_cols = [col for col in dataset.columns if col.startswith("feat")]
    feat_array = dataset[feat_cols]

    labels = dataset[included_modes]
    print("labels", labels.shape, "features", feat_array.shape)

    if return_normed:
        # normalize
        feat_mean, feat_std = feat_array.mean(), feat_array.std()
        feat_array_normed = (feat_array - feat_mean) / feat_std
        return feat_array_normed, labels, (feat_mean, feat_std)

    return feat_array, labels
