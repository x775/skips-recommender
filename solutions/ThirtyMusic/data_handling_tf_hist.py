import math
import pandas as pd

def get_subsessions(sessions):
    
    def generate_subsessions(pairs):
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

    tracks = sessions.track_idxs.values.tolist()
    tags = sessions.tags_idxs.values.tolist()
    histories = sessions.history.values.tolist()
    pairs = list(zip(tracks, tags, histories))

    return generate_subsessions(pairs)