import sys

def run():
    model = sys.argv[-1]
    if 'lastfm' or '30music' in model:
        if model == "lastfm-stabr":
            import solutions.STABR.STABR_tf.train_and_eval_tf as stabr
        elif model == 'lastfm-sabr':
            import solutions.STABR.STABR_tf.train_and_eval_songs_tf as stabr
        elif model == 'lastfm-skip':
            import solutions.STABR.STABR_skip.train_and_eval_tf as stabr
        elif model == 'lastfm-hist':
            import solutions.STABR.STABR_history_concat.train_and_eval_tf as stabr
        elif model == 'lastfm-hist-aslm':
            import solutions.STABR.STABR_history_aslm.train_and_eval_tf as stabr
        elif model == '30music-stabr':
            import solutions.ThirtyMusic.train_and_eval_tf as stabr
        elif model == '30music-sabr':
            import solutions.ThirtyMusic.train_and_eval_songs_tf as stabr
        elif model == '30music-skip':
            import solutions.ThirtyMusic.train_and_eval_tf_skips as stabr
        elif model == '30music-hist':
            import solutions.ThirtyMusic.train_and_eval_tf_hist as stabr
        elif model == '30music-hist-aslm':
            import solutions.ThirtyMusic.train_and_eval_tf_hist_aslm as stabr
        elif model == "30music-attless":
            import solutions.ThirtyMusic.train_and_eval_tf_attless as stabr
        else:
            print("Please enter a correct model name.")
            exit()
        stabr.run_train_test()
    else:
        print("Please enter a correct model name.")
        exit()

        
if __name__ == "__main__":
    run()
