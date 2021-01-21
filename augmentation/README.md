# Data Augmentation and Pre-processing

We employ two distinct datasets in order to (a) reproduce the results attained by Sachdeva et al.'s STABR architecture [0] and explore our proposed extensions; and (b) investigate the merits of complex architectures in terms of generalisable music recommendation.

We are working with the following datasets:

* Lastfm-1K, originally compiled by Òscar Celma [1], contains approximately $20$ million listening events separated into listening sessions submitted by $992$ distinct Last.fm users between 2005 and 2009.
* 30Music, compiled by Turrin et al. [2] as part of RecSys 2015, contains approximately 5.6M tracks, approximately 31M listening events separated into listening sessions submitted by $45$K distinct Last.fm users between 2014 and 2015.

Kindly refer to our paper for specific pre-processing steps.

The raw data is available for download at:

* http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html for Lastfm-1k;
* and http://recsys.deib.polimi.it/datasets/ for 30Music.




[0] https://dl.acm.org/doi/10.1145/3240323.3240397
[1] https://www.springer.com/gp/book/9783642132865
[2] http://ceur-ws.org/Vol-1441/recsys2015_poster13.pdf