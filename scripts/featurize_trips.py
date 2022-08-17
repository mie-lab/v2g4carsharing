from v2g4carsharing.mode_choice_model.features import ModeChoiceFeatures

if __name__ == "__main__":
    feat_collector = ModeChoiceFeatures()
    feat_collector.add_all_features()
    feat_collector.save("../data/mobis")
    # basic_trip_features = ["length"]
    #
    # time_features = ["started_at_hour", "started_at_day", "finished_at_hour"]  # TODO
    # # only keep important modes
    # drop_feats = [
    #     "prev_mode_Mode::Boat",
    #     "prev_mode_Mode::Cablecar",
    #     "prev_mode_Mode::MotorbikeScooter",
    #     "prev_mode_Mode::RidepoolingPikmi",
    #     "prev_mode_Mode::Ski",
    #     "prev_mode_Mode::TaxiUber",
    # ]
    # prev_mode_features = [c for c in one_hot_prev_mode.columns if c not in drop_feats]  # TODO

    # # TODO: load self.legs dataset
    # dataset = add_survey_features(dataset, survey_features)
    # dataset = add_time_features(dataset)
    # dataset = add_prev_mode_feature(dataset)

    # label_col = "mode"
    # dataset = dataset[survey_features + time_features + basic_trip_features + list(prev_mode_features) + [label_col]]
