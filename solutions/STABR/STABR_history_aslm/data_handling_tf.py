import math
import pandas as pd

def get_split(user_data, session_data, train_split, validation_split=None):
    def split_user_sids(sids):
        split_index = math.ceil(len(sids) * train_split)
        if validation_split:
            val_split_index = math.ceil(len(sids) * (train_split + validation_split))
            return sids[:split_index], sids[split_index:val_split_index], sids[val_split_index:]
        else:
            return sids[:split_index], sids[split_index:]

    def get_subsessions(pairs):
        subsessions = {}
        counter = 0
        for session in pairs:
            session_tracks = session[0]
            session_tags = session[1]
            session_history = session[2]
            for i in range(1, len(session_tracks)):
                subsessions[counter] = {"tracks" : session_tracks[:i+1],
                                        "tags" : session_tags[:i+1],
                                        "history" : session_history}
                counter += 1

        return pd.DataFrame.from_dict(subsessions, orient="index")


    split = user_data["sessions_subset"].apply(split_user_sids)
    if validation_split:
        train_sids = [item for sublist in split.tolist() for item in sublist[0]]
        val_sids = [item for sublist in split.tolist() for item in sublist[1]]
        test_sids = [item for sublist in split.tolist() for item in sublist[2]]
    else:        
        train_sids = [item for sublist in split.tolist() for item in sublist[0]]
        test_sids = [item for sublist in split.tolist() for item in sublist[1]]

    train_session_data = session_data.loc[train_sids]
    test_session_data = session_data.loc[test_sids]

    train_tracks = train_session_data.track_idxs.values.tolist()
    train_tags = train_session_data.tags_idxs.values.tolist()
    train_histories = train_session_data.history.values.tolist()
    train_pairs = list(zip(train_tracks, train_tags, train_histories))

    test_tracks = test_session_data.track_idxs.values.tolist()
    test_tags = test_session_data.tags_idxs.values.tolist()
    test_histories = test_session_data.history.values.tolist()
    test_pairs = list(zip(test_tracks, test_tags, test_histories))

    if validation_split:
        val_session_data = session_data.loc[val_sids]
        val_tracks = val_session_data.track_idxs.values.tolist()
        val_tags = val_session_data.tags_idxs.values.tolist()
        val_histories = val_session_data.history.values.tolist()
        val_pairs = list(zip(val_tracks, val_tags, val_histories))
        return train_session_data, get_subsessions(train_pairs), get_subsessions(val_pairs), get_subsessions(test_pairs)
    else:
        return train_session_data, get_subsessions(train_pairs), get_subsessions(test_pairs)
