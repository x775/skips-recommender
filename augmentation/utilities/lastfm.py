import json
import time
import urllib3
import logging
import requests
from typing import List
from bs4 import BeautifulSoup
from logger import setup_loggers

print("Creating Last.fm logger")
lastfm_logger = setup_loggers("lastfm", "lastfm.log",logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
lastfm_logger.info("Created Last.fm logger.")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def scrape_tags(url: str, proxy: dict) -> List:
    try:
        r = requests.get("https://last.fm/music/" + url + "/+tags", proxies=proxy, 
                         allow_redirects=True, verify=False)
    except requests.exceptions.Timeout:
        lastfm_logger.error("Unable to access last.fm's servers due to timeout. Sleeping 60.")
        time.sleep(60)
        return scrape_tags(url, proxy)
    
    soup = BeautifulSoup(r.content, "lxml")
    tags = soup.find_all("a", {"class" : "link-block-target"})
    tags = [tag["href"].replace("/tag/","").replace("+", " ") for tag in tags 
            if "/tag/" in tag["href"]]

    return tags


def get_info(track: str, artist: str, api_key: str, proxy: dict):
    try:
        r = requests.get(
            "http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={}&artist={}&track={}&format=json"\
            .format(api_key, artist.replace(" ", "+").replace("&", "and"), 
                    track.replace(" ", "+").replace("&", "and")), proxies=proxy)
    except requests.exceptions.Timeout:
        lastfm_logger.error("Unable to access last.fm's servers due to timeout. Sleeping 60.")
        time.sleep(60)
        return get_info(artist, track, api_key, proxy)

    try:    
        content = json.loads(r.content)
    except json.JSONDecodeError:
        # This happens only if <xml> is returned, which appears to only
        # happen when a track has not been found. In other words, the 
        # track has existed at one point, but has later been removed.
        # We thus return None and skip the track in question.
        print("A JSONDecodeError was caught for {} by {}.".format(track, artist))
        return None

    # Handle instances in which there is no available information.
    if "error" in list(content.keys()):
        return None
    
    info = {}
    info["total_listeners"] = content["track"]["listeners"]
    info["total_playcount"] = content["track"]["playcount"]
    info["duration_lastfm"] = content["track"]["duration"]
    info["tags"] = [tag["name"].lower() for tag in 
                    content["track"]["toptags"]["tag"]]
    # Rather than saving the entire URL, we only save the unique part.
    info["lastfm_url"] = content["track"]["url"].split("/music/")[1]
    
    lastfm_logger.info("Added {}".format(track))
    
    return info
