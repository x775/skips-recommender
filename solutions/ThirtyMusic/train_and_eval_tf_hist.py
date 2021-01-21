import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from solutions.ThirtyMusic import stabr_history
from solutions.ThirtyMusic import data_handling_tf_hist


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
    # Manually define relevant hyperparameters.
    EPOCHS = 100
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 100
    VAL_BATCH_SIZE = 100
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.15
    TRAIN_TEST_SPLIT = 0.7
    NP_RANDOM_SEED = 0
    VALIDATION_SIZE = 0.1
    SONG_EMBEDDING_SIZE = 75
    TAG_EMBEDDING_SIZE = 100
    TRACK_HISTORY_HIDDEN_SIZE = 64
    TAG_HISTORY_HIDDEN_SIZE = 64
    HISTORY_SIZE = 10
    DROPOUT = 0.1
    V_LAYER_SIZE = 25
    EVAL_MODEL_PATH = "models{0}stabr_25.ckpt".format(os.sep)
    
    ABSOLUTE_PATH = str(Path(__file__).resolve().parents[0].absolute())

                    #      TRACKS, TAGS
    training_values = {0: (216054, 54170),
                       1: (210736, 52441),
                       2: (200469, 50504),
                       3: (207245, 51153),
                       4: (218665, 52616)}
    
    testing_values = {0: (16305, 7620),
                      1: (14026, 6898),
                      2: (14472, 6682),
                      3: (15146, 7108),
                      4: (20318, 8607)}

    tf.config.optimizer.set_jit(True)
    
    # Since the data is already separated into training and testing splits, we
    # do not need to create these again. Instead, we just need to read each as 
    # a train_set and test_set before creating our DataLoaders.
    for index in range(3):
        TRAINING_SET = "{0}{1}data{1}30music_history{1}sessions_history_30music_training_{2}.json"\
                       .format(ABSOLUTE_PATH, os.sep, index)
        TESTING_SET = "{0}{1}data{1}30music_history{1}sessions_history_30music_testing_{2}.json"\
                      .format(ABSOLUTE_PATH, os.sep, index)
        TOTAL_SONGS, TOTAL_TAGS = training_values[index]

        training_set = pd.read_json(TRAINING_SET, orient="index")

        # Split training data into training and validation sets
        n_samples = len(training_set)
        sidx = np.arange(n_samples, dtype="int32")
        np.random.seed(NP_RANDOM_SEED)
        np.random.shuffle(sidx)
        n_train = int(np.round(n_samples * (1 - VALIDATION_SIZE)))
        train_data = training_set.iloc[sidx[:n_train]]
        val_data = training_set.iloc[sidx[n_train:]]

        train_sessions = data_handling_tf_hist.get_subsessions(train_data)
        val_sessions = data_handling_tf_hist.get_subsessions(val_data)
        test_sessions = data_handling_tf_hist.get_subsessions(pd.read_json(TESTING_SET, orient="index"))

        # Create target columns
        train_sessions["target"] = train_sessions["tracks"].apply(lambda idxs: idxs[-1])
        val_sessions["target"] = val_sessions["tracks"].apply(lambda idxs: int(idxs[-1]))
        test_sessions["target"] = test_sessions["tracks"].apply(lambda idxs: idxs[-1])
        train_sessions["target"] = train_sessions["target"] - 1
        val_sessions["target"] = val_sessions["target"] - 1
        test_sessions["target"] = test_sessions["target"] - 1

        # Remove target column track id and tag ids from input
        train_sessions["tracks"] = train_sessions["tracks"].apply(lambda idxs: idxs[:-1])
        train_sessions["tags"] = train_sessions["tags"].apply(lambda idxs: idxs[:-1])
        val_sessions["tracks"] = val_sessions["tracks"].apply(lambda idxs: idxs[:-1])
        val_sessions["tags"] = val_sessions["tags"].apply(lambda idxs: idxs[:-1])
        test_sessions["tracks"] = test_sessions["tracks"].apply(lambda idxs: idxs[:-1])
        test_sessions["tags"] = test_sessions["tags"].apply(lambda idxs: idxs[:-1])

        # Create Datasets for each set
        train_records = [tuple(row) for row in train_sessions.to_numpy()]
        val_records = [tuple(row) for row in val_sessions.to_numpy()]
        test_records = [tuple(row) for row in test_sessions.to_numpy()]

        # Create dictionaries for fetching histories
        sessions_train_hist = pd.read_json(TRAINING_SET, orient="index")
        session_train_idxs = sessions_train_hist.index.values.tolist()
        session_train_tracks = sessions_train_hist.track_idxs.values.tolist()
        session_train_tags = sessions_train_hist.tags_idxs.values.tolist()
        session_train_dict = dict(zip(session_train_idxs, zip(session_train_tracks, session_train_tags)))
        
        sessions_test_hist = pd.read_json(TESTING_SET, orient="index")
        session_test_idxs = sessions_test_hist.index.values.tolist()
        session_test_tracks = sessions_test_hist.track_idxs.values.tolist()
        session_test_tags = sessions_test_hist.tags_idxs.values.tolist()
        session_test_dict = dict(zip(session_test_idxs, zip(session_test_tracks, session_test_tags)))
        session_dict = {**session_train_dict, **session_test_dict}

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
        
        def val_generator():
            for record in val_records:
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

        val_dataset = tf.data.Dataset.from_generator(train_generator, 
            output_types=((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
            output_shapes=(((None,), (None, None,), (None, None,), (None, None, None,)), ()))
        val_dataset = val_dataset.padded_batch(VAL_BATCH_SIZE,
            padded_shapes=(([None], [None, None], [None, None], [None, None, None]), []))
        
        test_dataset = tf.data.Dataset.from_generator(test_generator, 
            output_types=((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
            output_shapes=(((None,), (None, None,), (None, None,), (None, None, None,)), ()))
        test_dataset = test_dataset.padded_batch(TEST_BATCH_SIZE,
            padded_shapes=(([None], [None, None], [None, None], [None, None, None]), []))

        print("Created tf datasets...")

        stabr = stabr_history.STABR(TOTAL_SONGS + 1, SONG_EMBEDDING_SIZE, TOTAL_TAGS + 1,
                                    TAG_EMBEDDING_SIZE, HIDDEN_SIZE, TRACK_HISTORY_HIDDEN_SIZE, 
                                    TAG_HISTORY_HIDDEN_SIZE, DROPOUT, V_LAYER_SIZE)
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        stabr.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])
            
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            "stabr_hist_30music_models/stabr_hist_30music_set_" + str(index) + "_{epoch}.ckpt",
            save_weights_only=True,
            period=1
        )

        print("Created network...")

        if "train" in sys.argv:
            stabr.fit(train_dataset, 
                      epochs=EPOCHS, 
                      verbose=1,
                      callbacks=[checkpoint_cb], 
                      validation_data=val_dataset, 
                      shuffle=False)
        elif "eval" in sys.argv:
            eval_model_path = "stabr_hist_30music_models/stabr_30music_set{}_{}.ckpt".format(index, sys.argv[2])
            for k in [1, 5, 10, 20, 30, 40, 50]:
                stabr_eval = stabr_history.STABR(TOTAL_SONGS + 1, 
                                                 SONG_EMBEDDING_SIZE, 
                                                 TOTAL_TAGS + 1,
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
            EPOCH = sys.argv[2]
            eval_model_path = "stabr_hist_30music_models/stabr_hist_30music_set{}_{}.ckpt".format(index, EPOCH)
            init_data = iter(train_dataset).next()
            inp, target = init_data
            stabr.train_on_batch(inp, target)
            stabr.load_weights(eval_model_path)
            stabr.fit(train_dataset, 
                      epochs=EPOCHS, 
                      verbose=1, 
                      callbacks=[checkpoint_cb], 
                      shuffle=False, 
                      initial_epoch=EPOCH)
        else:
            print("Missing arguments.")
