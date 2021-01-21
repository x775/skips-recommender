# Data Augmentation and Pre-processing

This folder contains all relevant code to augmenting our data (both the 1K and 1B set).

`combine.ipynb` contains all code required to create `.json` files for our models.

`utilities` contains all helper functions for the scrapers.

> **NOTE** that a `keys.json` file is expected in the utilities folder. As this file contains API keys, we have not included it. Please ensure to include a Last.fm Developer API key, a Spotify Client ID, and a Spotify Client Secret. These can be obtained for free from https://www.last.fm/api/ and https://developer.spotify.com/dashboard/, respectively.

We employ two distinct datasets in order to (a) reproduce the results attained by Sachdeva et al.'s STABR architecture [0] and explore our proposed extensions; and (b) investigate the merits of complex architectures in terms of generalisable music recommendation. Please refer to our paper for a high-level description of the general pre-processing steps.

We are working with the following datasets:

* Lastfm-1K, originally compiled by Òscar Celma [1], contains approximately 20 million listening events separated into listening sessions submitted by 992 distinct Last.fm users between 2005 and 2009.
* 30Music, compiled by Turrin et al. [2] as part of RecSys 2015, contains approximately 5.6M tracks, approximately 31M listening events separated into listening sessions submitted by 45K distinct Last.fm users between 2014 and 2015.

The raw data is available for download at:

* http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html for Lastfm-1k;
* and http://recsys.deib.polimi.it/datasets/ for 30Music.



<small>

[0] https://dl.acm.org/doi/10.1145/3240323.3240397

[1] https://www.springer.com/gp/book/9783642132865

[2] http://ceur-ws.org/Vol-1441/recsys2015_poster13.pdf
</small>