import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from solutions.STABR.STABR_history_concat import stabr_history
from solutions.STABR.STABR_history_concat import data_handling_tf


def eval_mrr(network, test_data, k=10):
    cumulative_rank = 0
    total = 0
    for step, data in enumerate(test_data):
        inp, targets = data
        logits = network(inp, training=False)
        log_probs = tf.nn.log_softmax(logits, axis=1)

        for i in range(targets.shape[0]):
            _, output_idxs = tf.nn.top_k(log_probs[i], k=k)
            total += 1
            target = targets[i]
            if target in output_idxs:
                rank = np.where(output_idxs.numpy() == target)[0][0]
                cumulative_rank += 1 / (rank + 1)
        
        if step % 100 == 99:
            print("MRR @ step {}: {}".format(step, cumulative_rank / total))
    
    mrr = cumulative_rank / total
    print("Mean Reciprocal Rank @ {}: {}".format(k, mrr))



def run_train_test():
    ABSOLUTE_PATH = str(Path(__file__).resolve().parents[2].absolute())
    EPOCHS = 60
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 100
    HIDDEN_SIZE = 96
    LEARNING_RATE = 0.2
    TRAIN_TEST_SPLIT = 0.7
    VALIDATION_SPLIT = 0.09
    SONG_EMBEDDING_SIZE = 50
    TAG_EMBEDDING_SIZE = 100
    TRACK_HISTORY_HIDDEN_SIZE = 64
    TAG_HISTORY_HIDDEN_SIZE = 64
    HISTORY_SIZE = 10
    DROPOUT = 0.1
    V_LAYER_SIZE = 50

    # Load all data as dataframes.
    with open(ABSOLUTE_PATH + "{0}data{0}history{0}sessions_10sessions_2plays_history_tf.json".format(os.sep), "r") as source:
        sessions = pd.read_json(source, orient="index")

    with open(ABSOLUTE_PATH + "{0}data{0}history{0}tracks_10sessions_2plays_history_tf.json".format(os.sep), "r") as source:
        songs = pd.read_json(source, orient="index")
        songs.index = songs.index + 1

    with open(ABSOLUTE_PATH + "{0}data{0}history{0}users_10sessions_2plays_history_tf.json".format(os.sep), "r") as source:
        users = pd.read_json(source, orient="index")

    print("Created dataframes...")

    # Grab total number of tags.
    with open(ABSOLUTE_PATH + "{0}data{0}history{0}tags_10sessions_2plays_history_tf.json".format(os.sep), "r") as source:
        total_tags = len(json.load(source))

    # Create training and testing splits. 
    _, train_sessions, _, test_sessions = data_handling_tf.get_split(users, sessions, TRAIN_TEST_SPLIT, VALIDATION_SPLIT)
    # create target columns
    train_sessions["target"] = train_sessions["tracks"].apply(lambda idxs: idxs[-1])
    test_sessions["target"] = test_sessions["tracks"].apply(lambda idxs: idxs[-1])
    train_sessions["target"] = train_sessions["target"] - 1
    test_sessions["target"] = test_sessions["target"] - 1
    #remove target column track id and tag ids from input
    train_sessions["tracks"] = train_sessions["tracks"].apply(lambda idxs: idxs[:-1])
    train_sessions["tags"] = train_sessions["tags"].apply(lambda idxs: idxs[:-1])
    test_sessions["tracks"] = test_sessions["tracks"].apply(lambda idxs: idxs[:-1])
    test_sessions["tags"] = test_sessions["tags"].apply(lambda idxs: idxs[:-1])
    # Create Datasets for each set
    train_records = [tuple(row) for row in train_sessions.to_numpy()]
    test_records = [tuple(row) for row in test_sessions.to_numpy()]
    # Create dictionaries for fetching histories
    session_idxs = sessions.index.values.tolist()
    session_tracks = sessions.track_idxs.values.tolist()
    session_tags = sessions.tags_idxs.values.tolist()
    session_dict = dict(zip(session_idxs, zip(session_tracks, session_tags)))

    def train_generator():
        for record in train_records:
            tracks = record[0]
            tags = record[1]
            history = record[2]
            target = record[-1]
            if len(history) > 0:
                tracks_history = [session_dict[i][0] for i in history[-HISTORY_SIZE:]]
                session_len = max(map(len, tracks_history))
                tracks_history = [x + [0]*(session_len - len(x)) for x in tracks_history]
                tags_history = [session_dict[i][1] for i in history[-HISTORY_SIZE:]]
                tags_len = max([len(tags) for songs in tags_history for tags in songs])
                tags_history = [pad_sequences(songs, tags_len, padding="post").tolist() for songs in tags_history]
                for session in tags_history:
                    for i in range(session_len - len(session)):
                        session.append([0] * tags_len)
            else:
                tracks_history = [[0]]
                tags_history = [[[0]]]
            yield (tracks, tracks_history, tags, tags_history), target

    def test_generator():
        for record in test_records:
            tracks = record[0]
            tags = record[1]
            history = record[2]
            target = record[-1]
            if len(history) > 0:
                tracks_history = [session_dict[i][0] for i in history[-HISTORY_SIZE:]]
                session_len = max(map(len, tracks_history))
                tracks_history = [x + [0]*(session_len - len(x)) for x in tracks_history]
                tags_history = [session_dict[i][1] for i in history[-HISTORY_SIZE:]]
                tags_len = max([len(tags) for songs in tags_history for tags in songs])
                tags_history = [pad_sequences(songs, tags_len, padding="post").tolist() for songs in tags_history]
                for session in tags_history:
                    for i in range(session_len - len(session)):
                        session.append([0] * tags_len)
            else:
                tracks_history = [[0]]
                tags_history = [[[0]]]
            yield (tracks, tracks_history, tags, tags_history), target

    train_dataset = tf.data.Dataset.from_generator(train_generator, 
        output_types=((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
        output_shapes=(((None,), (None, None,), (None, None,), (None, None, None,)), ()))
    train_dataset = train_dataset.padded_batch(BATCH_SIZE,
        padded_shapes=(([None], [None, None], [None, None], [None, None, None]), []))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_generator(test_generator, 
        output_types=((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
        output_shapes=(((None,), (None, None,), (None, None,), (None, None, None,)), ()))
    test_dataset = test_dataset.padded_batch(TEST_BATCH_SIZE,
        padded_shapes=(([None], [None, None], [None, None], [None, None, None]), []))

    print("Created tf datasets...")

    # https://www.tensorflow.org/xla
    tf.config.optimizer.set_jit(True)

    stabr = stabr_history.STABR(len(songs) + 1, 
                                SONG_EMBEDDING_SIZE, 
                                total_tags + 1,
                                TAG_EMBEDDING_SIZE, 
                                HIDDEN_SIZE, 
                                TRACK_HISTORY_HIDDEN_SIZE, 
                                TAG_HISTORY_HIDDEN_SIZE, 
                                DROPOUT, 
                                V_LAYER_SIZE)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    stabr.compile(optimizer=optimizer, 
                  loss=loss_fn, 
                  metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])
        
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "models/stabr_history_concat_{epoch}.ckpt",
        save_weights_only=True,
        period=1
    )

    print("Created network...")

    if "train" in sys.argv:
        stabr.fit(train_dataset, epochs=EPOCHS, verbose=1, callbacks=[checkpoint_cb], shuffle=False)
    elif "eval" in sys.argv:
        eval_model_path = "models{0}stabr_history_concat_{1}.ckpt".format(os.sep, sys.argv[2])
        for k in [10, 20, 30, 40, 50]:
            stabr_eval = stabr_history.STABR(len(songs) + 1, 
                                             SONG_EMBEDDING_SIZE, 
                                             total_tags + 1,
                                             TAG_EMBEDDING_SIZE, 
                                             HIDDEN_SIZE, 
                                             TRACK_HISTORY_HIDDEN_SIZE, 
                                             TAG_HISTORY_HIDDEN_SIZE, 
                                             DROPOUT, 
                                             V_LAYER_SIZE)
            stabr_eval.compile(optimizer=optimizer, 
                               loss=loss_fn, 
                               metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)])
            init_data = iter(train_dataset).next()
            inp, target = init_data
            stabr_eval.train_on_batch(inp, target)
            stabr_eval.load_weights(eval_model_path)
            if sys.argv[3] == "hr":
                stabr_eval.evaluate(test_dataset)
            elif sys.argv[3] == "mrr":
                eval_mrr(stabr_eval, test_dataset, k)
    elif "checkpoint" in sys.argv:
        EPOCH = int(sys.argv[2])
        eval_model_path = "models/stabr_history_concat_{}.ckpt".format(EPOCH)
        init_data = iter(train_dataset).next()
        inp, target = init_data
        stabr.train_on_batch(inp, target)
        stabr.load_weights(eval_model_path)
        stabr.fit(train_dataset, epochs=EPOCHS, verbose=1, callbacks=[checkpoint_cb], shuffle=False, initial_epoch=EPOCH)
    else:
        print("Missing arguments.")
