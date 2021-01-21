import csv
import sys
import json
import time
import logging
import requests
from enum import Enum
from queue import Queue
from typing import Tuple
from threading import Thread
from utilities import lastfm
from utilities import spotify
from datetime import datetime


class TrackAugmenter(Thread):
    def __init__(self, api_token: str, proxy: dict, spotify_auth: str=None, next_proxy=None):
        super(TrackAugmenter, self).__init__()
        self.api_token = api_token
        self.proxy = proxy
        self.last_token_refresh = datetime.now()
        self.spotify_auth = spotify_auth
        self.heartbeat = datetime.now()
        self.minutes_till_death = 1
        self.secondary = api_token
        self.thirdiary = spotify_auth
        self.next_proxy = next_proxy
    
    def has_crashed(self) -> bool:
        if not self.is_alive():
            return True
        return (datetime.now() - self.heartbeat).seconds / 60 > self.minutes_till_death


lastfm_track_queue = Queue()
lastfm_augmented_tracks = Queue()
tracks_without_tags = Queue()


class LastfmAPIAugmenter(TrackAugmenter):
    def __init__(self, api_token: str, proxy: dict):
        super(LastfmAPIAugmenter, self).__init__(api_token, proxy)

    def run(self):
        while not lastfm_track_queue.empty():
            self.heartbeat = datetime.now()
            # Retrieve triple of (track_id, track_name, artist_name)
            track: Tuple = lastfm_track_queue.get()

            # Call API and retrieve total listeners, playcount, duration, tags and url
            data: dict = lastfm.get_info(track[1], track[2], self.api_token, self.proxy)
    
            # Last.fm might return None
            if data:
                data["available"] = 1
                data["track_id"] = track[0]
                # Check if any tags were returned by API
                if len(data["tags"]) == 0:
                    tracks_without_tags.put(data)
                else:
                    lastfm_augmented_tracks.put(data)
                print("Completed Last.fm search for {}".format(track[1]))
            else:
                data = {"total_listeners": "",
                        "total_playcount": "",
                        "duration_lastfm": "",
                        "tags": "",
                        "lastfm_url": "",
                        "available": 0,
                        "track_id": track[0]}
                lastfm_augmented_tracks.put(data)
                print("No information available for {}".format(track[1]))

            time.sleep(1)
            lastfm_track_queue.task_done()


class LastfmTagScraper(TrackAugmenter):
    def __init__(self, proxy: dict):
        super(LastfmTagScraper, self).__init__(None, proxy)

    def run(self):
        while True:
            self.heartbeat = datetime.now()
            # Get track missing tag(s)
            data = tracks_without_tags.get()
            
            # Lookup missing tags
            try:
                data["tags"] = lastfm.scrape_tags(data["lastfm_url"], self.proxy)
            except requests.exceptions.TooManyRedirects:
                print("TooManyRedirects caught for {}, leaving data empty.".format(data["track_id"]))
                data["tags"] = ""
           
            # We now have all tags for track
            lastfm_augmented_tracks.put(data)

            print("Scraped tag for track_id {}".format(data["track_id"]))
            time.sleep(1)
            tracks_without_tags.task_done()


class LastfmWriter(Thread):
    def __init__(self):
        super(LastfmWriter, self).__init__()

    def run(self):
        while True:
            # Retrieve fully augmented track data
            data: dict = lastfm_augmented_tracks.get()
            with open("lastfm.csv", "a+") as f:
                f.write(",".join(map(str, data.values())) + "\n")
                lastfm_augmented_tracks.task_done()


spotify_track_queue = Queue()
tracks_to_featurify = Queue()
spotify_augmented_tracks = Queue()


class SpotifySearchAugmenter(TrackAugmenter):
    def __init__(self, api_token: str, proxy: dict, spotify_auth: str, next_proxy):
        super(SpotifySearchAugmenter, self).__init__(api_token, proxy, spotify_auth, next_proxy)

    def run(self):
        counter = 0
        while not spotify_track_queue.empty():
            self.heartbeat = datetime.now()
            if counter > 5:
                print("Encountered 5 tracks without any data in a row, sleeping 30.")
                time.sleep(30)
                counter = 0

            # Retrieve triple of (track_id, track_name, artist_name)
            track = spotify_track_queue.get()
            
            # Get release date and Spotify ID
            data: dict = spotify.get_id(track[1], track[2], self, next_proxy)
            data["track_id"] = track[0]

            # No features will be available if Spotify ID is 0
            if data["spotify_id"]:
                tracks_to_featurify.put(data)
                counter = 0
            else:
                # Add empty kv pairs to enable empty CSV commas
                for i in range(13):
                    data[i] = ""
                spotify_augmented_tracks.put(data)
                counter += 1
            
            print("Completed Spotify search for {}".format(track[1]))
            time.sleep(1)
            spotify_track_queue.task_done()


class SpotifyFeaturesAugmenter(TrackAugmenter):
    def __init__(self, api_token: str, proxy: dict, spotify_auth: str, next_proxy):
        super(SpotifyFeaturesAugmenter, self).__init__(api_token, proxy, spotify_auth, next_proxy)

    def run(self):
        while True:
            self.heartbeat = datetime.now()
            # Construct list of 100 songs to featurify
            batch = []
            while len(batch) < 100:
                batch.append(tracks_to_featurify.get())

            # Request features for batch of songs
            batch = spotify.get_audio_features(batch, self, next_proxy)

            # Add each augmented track to queue
            for track in batch:
                spotify_augmented_tracks.put(track)

            print("Completed audio features for 100 tracks. Sleeping 5.")
            time.sleep(5)
            tracks_to_featurify.task_done()


class SpotifyWriter(Thread):
    def __init__(self):
        super(SpotifyWriter, self).__init__()

    def run(self):
        while True:
            # Retrieve features from Spotify
            data: dict = spotify_augmented_tracks.get()
            with open("spotify.csv", "a+") as f:
                f.write(",".join(map(str, data.values())) + "\n")
                spotify_augmented_tracks.task_done()


class RasAlGhul(Thread):
    def __init__(self, threads_dict: dict):
        super(RasAlGhul, self).__init__()
        self.threads_dict = threads_dict
        self.original_count = {}
        for k, v in threads_dict.items():
            self.original_count[k] = len(v)

    def run(self):
        while True:
            for group, threads in self.threads_dict.items():
                if group == ThreadType.WATCHER or group == None or group == ThreadType.WRITER:
                    continue
                for thread in threads:
                    if thread.has_crashed():
                        # restart thread
                        print("sippin from lazarus pit")
                        start_thread(thread_factory(group, next_proxy(), thread.secondary, thread.thirdiary), group)
            time.sleep(2)


STATIC_FOLDER = "data/"
def read_file(filename: str, capped_after: int=-1) -> dict:
    content = {}
    with open(STATIC_FOLDER + filename, "r") as source:
        for i, line in enumerate(source):
            if capped_after > 0 and capped_after <= i:
                break
            row = [l.strip() for l in line.split("\t")]
            content[row[0]] = tuple(row[1:])
    return content


def read_csv_file(filename: str) -> list:
    with open(STATIC_FOLDER + filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        return list(reader)


def convert_to_lastfmdict(filename: str) -> dict:
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        existing = {}
        for entry in reader:
            existing[entry[-1]] = True  
        return existing


def convert_to_spotifydict(filename: str) -> dict:
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        existing = {}
        for entry in reader:
            existing[entry[2]] = True  
        return existing


class ThreadType(Enum):
    LASTFM_TAGS = 0
    LASTFM_AUGMENTER = 1 
    LASTFM_TAGS_FREE = 4
    SPOTIFY_SEARCH = 2 
    SPOTIFY_AUGMENTER = 3
    WATCHER = 9001
    WRITER = -500


def thread_factory(thread_type: ThreadType, proxy: str, secondary: str=None, thirdiary: str=None) -> Thread:
    t = None
    if thread_type == ThreadType.LASTFM_TAGS:
        t = LastfmTagScraper(proxy)
        t.minutes_till_death = 5
    elif thread_type == ThreadType.LASTFM_AUGMENTER:
        t = LastfmAPIAugmenter(secondary, proxy)
        t.minutes_till_death = 3
    elif thread_type == ThreadType.LASTFM_TAGS_FREE:
        t = LastfmAPIAugmenter(secondary, proxy)
        t.minutes_till_death = 3
    elif thread_type == ThreadType.SPOTIFY_SEARCH:
        t = SpotifySearchAugmenter(secondary, proxy, spotify_auth=thirdiary, next_proxy=next_proxy)
        t.minutes_till_death = 3
    elif thread_type == ThreadType.SPOTIFY_AUGMENTER:
        t = SpotifyFeaturesAugmenter(secondary, proxy, spotify_auth=thirdiary, next_proxy=next_proxy)
        t.minutes_till_death = 8
    return t

threads_dict = {}
def start_thread(thread, group: ThreadType): 
    if group not in threads_dict:
        threads_dict[group] = []

    threads_dict[group].append(thread)

    thread.daemon = True
    thread.start()

with open("utilities/proxies.txt", "r") as f:
    proxies = [{"http": "http://" + proxy.strip() + ":8888", 
                "https": "https://" + proxy.strip() + ":8888"} 
                for proxy in f.readlines()]


with open("utilities/freeproxies.txt", "r") as f:
    freeproxies = [{"http": "http://" + proxy.strip(), 
                    "https": "https://" + proxy.strip(),
                    "available": 1} 
                    for proxy in f.readlines()]

fp_idx = 0
def next_free_proxy() -> dict:
    global fp_idx
    if freeproxies[p_idx % len(freeproxies)]["available"]:
        p = freeproxies[p_idx % len(freeproxies)]
        fp_idx += 1
        return p
    fp_idx += 1

p_idx = 0
def next_proxy() -> dict:
    global p_idx
    p = proxies[p_idx % len(proxies)]
    p_idx += 1
    return p

if __name__ == "__main__":
    # Insert Sentry here if needed.

    if "1b" in sys.argv:
        print("Reading data files...")
        artists = read_file("LFM-1b_artists.txt")
        tracks = read_file("LFM-1b_tracks.txt")

        print("Checking existing...")
        existing_lastfm = convert_to_lastfmdict("lastfm.csv")
        existing_spotify = convert_to_spotifydict("spotify.csv")

        print("Enqueueing tracks...")
        print(len(tracks))
        for key, value in tracks.items():
            track_id = str(key)
            track_name = value[0]
            artist_name = artists[value[1]][0]
            triple = (track_id , track_name, artist_name)
            # Add to queues only if missing.
            if track_id not in existing_lastfm:
                lastfm_track_queue.put(triple)
            if track_id not in existing_spotify:
                spotify_track_queue.put(triple)

    if "1k" in sys.argv:
        print("Reading data files...")
        unique_tracks = read_csv_file("unique-tracks.csv")
        
        print("Enqueueing tracks...")
        for index, entry in enumerate(unique_tracks):
            track_id = index
            track_name = entry[2]
            artist_name = entry[0]
            triple = (track_id, track_name, artist_name)
            spotify_track_queue.put(triple)
            lastfm_track_queue.put(triple)

    print("Enqueued {} tracks for Spotify.".format(spotify_track_queue.qsize()))
    print("Enqueued {} tracks for Lastfm.".format(lastfm_track_queue.qsize()))
    
    print("Reading API keys...")
    with open("utilities/keys.json", "r") as f:
        keys = json.load(f)
    
    lfm_idx = 0
    def next_last_fm() -> str:
        global lfm_idx
        key = keys[str(lfm_idx)]["API_KEY"]
        lfm_idx += 1
        return key

    s_idx = 0
    def next_spotify() -> str:
        global s_idx
        key = keys[str(s_idx)]["SPOTIFY_AUTH"]
        s_idx += 1
        return key

    num = len(proxies)
    num_free = len(freeproxies)
    num_augmenters = 2
    print("Starting {} Last.fm scraper threads, {} Spotify search threads, {} Spotify augmenter threads, {} Last.fm augmenter threads..."\
          .format(num, (num - num_augmenters), num_augmenters, num + num_free))

    print("Starting writers...")
    # Spawn 1 Last.fm + Spotify writer
    start_thread(LastfmWriter(), group=ThreadType.WRITER)
    start_thread(SpotifyWriter(), group=ThreadType.WRITER)

    for _ in range(num * 2):
        start_thread(thread_factory(ThreadType.LASTFM_TAGS, 
                                    proxy=next_proxy()), 
                                    group=ThreadType.LASTFM_TAGS)
    print("Started Last.fm tag scraper threads.")

    # Spawn `num_augmenters` Spotify feature threads
    for _ in range(num_augmenters):
        spotify_auth = next_spotify()
        proxy = next_proxy()
        start_thread(thread_factory(ThreadType.SPOTIFY_AUGMENTER,
                                    proxy=proxy, 
                                    secondary=spotify.refresh_token(spotify_auth, proxy, next_proxy), 
                                    thirdiary=spotify_auth), 
                                    group=ThreadType.SPOTIFY_AUGMENTER)
    print("Started Spotify augmentor threads.")
    
    # Spawn Last.fm augmenters
    for _ in range(num):
        start_thread(thread_factory(ThreadType.LASTFM_AUGMENTER, 
                                    proxy=next_proxy(), 
                                    secondary=next_last_fm()), 
                                    group=ThreadType.LASTFM_AUGMENTER)
    print("Started Last.fm augmentor threads.")

    # Spawn `num` - `num_augmenters` Spotify search threads (prevent running out of keys)
    for _ in range(num - num_augmenters):
        proxy = next_proxy()
        spotify_auth = next_spotify()
        start_thread(thread_factory(ThreadType.SPOTIFY_SEARCH,
                                    proxy=proxy, 
                                    secondary=spotify.refresh_token(spotify_auth, proxy, next_proxy), 
                                    thirdiary=spotify_auth), 
                                    group=ThreadType.SPOTIFY_SEARCH)
    print("Started Spotify search threads.")

    # Thread dedicated to resurrecting dead/crashed threads
    start_thread(RasAlGhul(threads_dict), ThreadType.WATCHER)
    print("🔥🔥🔥 LET IT RIP 🔥🔥🔥")

    spotify_track_queue.join()
    tracks_to_featurify.join()
    spotify_augmented_tracks.join()
    print("Completed Spotify queues.")

    lastfm_track_queue.join()
    tracks_without_tags.join()
    lastfm_augmented_tracks.join()
    print("Completed Last.fm queues.")
