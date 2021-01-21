import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime
from utilities.readwrite import write_list_to_disk
from utilities.logger import setup_loggers

print("Creating Spotify logger")
spotify_logger = setup_loggers("spotify", "spotify.log",logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
spotify_logger.info("Created Spotify logger.")


def refresh_token(spotify_auth: str, proxy: dict, next_proxy, retry=True):
    root = "https://accounts.spotify.com/api/token"
    headers = { "Authorization": "Basic " + spotify_auth }
    data = { "grant_type": "client_credentials" }
    
    if not retry:
        # If retry is not true, it means we have tried to get a token once
        # before but failed to attain one. As such, we try with a new proxy.
        proxy = next_proxy()
        spotify_logger.error("Retrying spotify_auth with new proxy. Old proxy {}, auth {}.".format(proxy, spotify_auth))
        return refresh_token(spotify_auth, proxy, next_proxy, retry=True)

    try:
        r = requests.post(root, headers=headers, data=data, proxies=proxy)
    except requests.exceptions.Timeout:
        spotify_logger.error("Unable to access Spotify's servers due to timeout. Sleeping 60. Proxy {}, auth {}.".format(proxy, spotify_auth))
        time.sleep(60)
        return refresh_token(spotify_auth, proxy, next_proxy, retry=False)

    try:
        content = json.loads(r.content)
    except json.JSONDecodeError:
        time.sleep(5)
        spotify_logger.error("A JSONDecodeError exception was caught, sleeping 5 and trying again. Content {}, auth {}, proxy {}.".format(r.content, spotify_auth, proxy))
        return refresh_token(spotify_auth, proxy, next_proxy, retry=False)
    
    if r.status_code == 429:
        try:
            seconds = int(content["Retry-After"])
        except KeyError:
            spotify_logger.error("Spotify's rate limit was hit. Missing retry-after. Sleeping 10. Proxy {}, auth {}.".format(proxy, spotify_auth))
            time.sleep(10)
            return refresh_token(spotify_auth, proxy, next_proxy, retry=False)
        spotify_logger.error("Spotify's rate limit was hit. Sleeping {}.".format(seconds))
        time.sleep(seconds)
        return refresh_token(spotify_auth, proxy, next_proxy, retry=False)
    elif r.status_code != 200:
        spotify_logger.error("Received HTTP-code {} with {} and proxy {} sleeping 5 and trying again.".format(r.status_code, spotify_auth, proxy))
        time.sleep(5)
        return refresh_token(spotify_auth, proxy, next_proxy, retry=False)

    try:
        return json.loads(r.content)["access_token"]
    except KeyError:
        spotify_logger.error("A KeyError exception was caught, sleeping 5 and trying again. Poxy {}, auth {}.".format(proxy, spotify_auth))
        time.sleep(5)
        return refresh_token(spotify_auth, proxy, next_proxy, retry=False)


def get_id(track: str, artist: str, owner, next_proxy):
    root = "https://api.spotify.com/v1/search?q=track%3A{}%20artist%3A{}&type=track"\
           .format(track.lower().replace(" ", "%20"), artist.lower().replace(" ", "%20"))

    headers = { "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer " + owner.api_token }
    
    # Verify that our token has not expired.
    if (datetime.now() - owner.last_token_refresh).seconds > 3550:
           owner.api_token = refresh_token(owner.spotify_auth, owner.proxy, next_proxy)
           owner.last_token_refresh = datetime.now()

    # We attempt to connect to Spotify's servers.
    try:
        r = requests.get(root, headers=headers, proxies=owner.proxy)
    except requests.exceptions.Timeout:
        spotify_logger.error("Unable to access Spotify's servers due to timeout. Sleeping 60.")
        time.sleep(60)
        return get_id(track, artist, owner, next_proxy)
    
    # We managed to reach the server and attain a response.
    try:
        content = json.loads(r.content)
    except json.JSONDecodeError:
        time.sleep(5)
        spotify_logger.error("A JSONDecodeError exception was caught, sleeping 5 and trying again.")
        return get_id(track, artist, owner, next_proxy)

    # If we have exceeded the rate limit.
    if r.status_code == 429:
        try:
            seconds = int(content["Retry-After"])
        except KeyError:
            spotify_logger.error("Spotify's rate limit was hit. Missing retry-after. Sleeping 10.")
            time.sleep(10)
            return get_id(track, artist, owner, next_proxy)
        
        # If we do get a Retry-After value returned, we sleep that accordingly.
        spotify_logger.error("Spotify's rate limit was hit. Sleeping {}.".format(seconds))
        time.sleep(seconds)
        return get_id(track, artist, owner, next_proxy)
    
    # Assuming no errors or connection issues, below return response content as JSON.
    def read_spotify_data(response):
        try:
            return json.loads(response.content)["tracks"]["items"]
        except (KeyError, json.JSONDecodeError) as e:
            spotify_logger.info("A {} exception was caught on loading spotify id. Content: {}".format(e, r.content))
            time.sleep(0.5)
            return None

    def find_accurate_song(target, data):
        similarity = []
        for song in data:
            name = song["name"]
            if name == target:
                return song
            similarity.append(SequenceMatcher(None, target, name).ratio())
        return data[np.argmax(similarity)]

    data = read_spotify_data(r)
    if data:
        matched = find_accurate_song(track, data)
        try:              
            return { "release_date": pd.to_datetime(matched["album"]["release_date"]).strftime("%Y-%m-%d"),
                     "spotify_id": matched["id"] }
        except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime:
            # This happens if the supplied release_date is missing (e.g. 0-01-01 00:00:00).
            # In such a situation, we simply catch the exception and pass it such that we
            # end up returning 0 and 0 accordingly.
            pass

    # If all else fails, we return the below.
    spotify_logger.info("Did not locate any results for {} by {}.".format(track, artist))
    return {"release_date": 0, "spotify_id": 0}


def get_audio_features(batch: list, owner, next_proxy):
    spotify_ids = []
    for entry in batch:
        spotify_ids.append(entry["spotify_id"])
    
    root = "https://api.spotify.com/v1/audio-features?ids={}"\
            .format("%2C".join(spotify_ids))

    headers = { "Accept" : "application/json",
                "Content-Type" : "application/json",
                "Authorization" : "Bearer " + owner.api_token }
    
    # Verify that our token has not expired.
    if (datetime.now() - owner.last_token_refresh).seconds > 3550:
           owner.api_token = refresh_token(owner.spotify_auth, owner.proxy, next_proxy)
           owner.last_token_refresh = datetime.now()
    
    # We attempt to connect to Spotify's servers.
    try:
        r = requests.get(root, headers=headers, proxies=owner.proxy)
    except requests.exceptions.Timeout:
        spotify_logger.error("Unable to access Spotify's servers due to timeout. Sleeping 60.")
        time.sleep(60)
        return get_audio_features(batch, owner, next_proxy)
    
    try:
        content = json.loads(r.content)
    except json.JSONDecodeError:
        time.sleep(5)
        spotify_logger.error("A JSONDecodeError exception was caught, sleeping 5 and trying again.")
        return get_audio_features(batch, owner, next_proxy)

    if r.status_code == 429:
        try:
            seconds = int(content["Retry-After"])
        except KeyError:
            spotify_logger.error("Spotify's rate limit was hit. Missing retry-after. Sleeping 10.")
            time.sleep(10)
            return get_audio_features(batch, owner, next_proxy) 
        
        spotify_logger.error("Spotify's rate limit was hit. Sleeping {}.".format(seconds))
        time.sleep(seconds)
        return get_audio_features(batch, owner, next_proxy) 
    elif r.status_code != 200:
        spotify_logger.error("Received HTTP-code {}, sleeping 5 and trying again.".format(r.status_code))
        time.sleep(5)
        return get_audio_features(batch, owner, next_proxy) 

    def update_empty(row):
        row.update({"acousticness" : "", "danceability" : "",
                    "energy" : "", "instrumentalness" : "",
                    "pitch_class" : "", "liveness" : "",
                    "loudness" : "", "mode" : "",
                    "speechiness" : "", "tempo" : "",
                    "time_signature" : "", "valence" : "",
                    "duration_spotify" : ""})

    try:
        data = content["audio_features"]
    except (KeyError, json.JSONDecodeError) as e:
        # If we receive a KeyError or JSONDecodeError despite the HTTP response being 200,
        # we dump all affected IDs to disk such that we can check later. We then iterate
        # through all rows and set their values to "" as to not break the format later. 
        spotify_logger.error("A {} exception was caught on loading audio features, saving affected ids...".format(e))
        for row in batch:
            update_empty(row)
        return batch
    
    for i, row in enumerate(batch):
        d = data[i]
        try:
            row.update({"acousticness": d["acousticness"],
                        "danceability": d["danceability"],
                        "energy": d["energy"],
                        "instrumentalness": d["instrumentalness"],
                        "pitch_class": d["key"],
                        "liveness": d["liveness"],
                        "loudness": d["loudness"],
                        "mode": d["mode"],
                        "speechiness": d["speechiness"],
                        "tempo": d["tempo"],
                        "time_signature": d["time_signature"],
                        "valence": d["valence"],
                        "duration_spotify": d["duration_ms"]})
        except TypeError:
            # If we somehow catch a TypeError, the dictionary is either empty or 
            # partially empty. Rather than trying every key, we set every value
            # to "" and write the affected ID to disk such that we can check later.
            spotify_logger.error("TypeError caught on Spotify ID: {}".format(row["spotify_id"]))
            write_list_to_disk([row["spotify_id"]], "missing_audio_features.csv")
            update_empty(row)
        
    return batch