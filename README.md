# Exploring Skips in Session-based Music Recommendation

In order to get started, run `pip install -r requirements.txt`.

We have tested on Ubuntu 18.04 (x86_64 Linux 4.15.0-88-generic), Windows 10, and macOS Catalina. GPU container images (tested with 20.03-tf2-py3) available at: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html

In order to train a model, run `python3 main.py train model-name`.

In order to evaluate a trained model, run `python3 main.py eval epoch-number model-name`.

`epoch-number` should be an int. Note that a model must train for at least one epoch before being able to be evaluated.

Valid options for model-name `model-name` are:
* lastfm-stabr
* lastfm-sabr
* lastfm-skip
* lastfm-hist
* lastfm-hist-aslm
* 30music-stabr
* 30music-sabr
* 30music-skip
* 30music-hist
* 30music-hist-aslm

In order to run `SKNN`, cd into `solutions/SKNN` and run `python3 sknn.py train` for training, and `python3 sknn.py eval` for evaluation. In order to run the SKNN-SKIPS variant, replace `sknn.py` with `sknn_skip.py`. This version is for the Lastfm-1K dataset only.
