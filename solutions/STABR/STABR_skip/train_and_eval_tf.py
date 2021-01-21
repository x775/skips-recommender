
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from solutions.STABR.STABR_skip import stabr_skip_tf
#from solutions.STABR.STABR_skip import stabr_skip_lstm
from solutions.STABR.STABR_skip import data_handling_tf


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
    EPOCHS = 100
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 100
    HIDDEN_SIZE = 96
    LEARNING_RATE = 0.2
    TRAIN_TEST_SPLIT = 0.7
    VALIDATION_SPLIT = 0.09
    SONG_EMBEDDING_SIZE = 50
    TAG_EMBEDDING_SIZE = 100
    SKIP_ACTIONS = 2
    SKIP_EMBEDDING_SIZE = 10
    DROPOUT = 0.1
    V_LAYER_SIZE = 75

    #LSTM:
    #HIDDEN_SIZE = 128
    #LEARNING_RATE = 0.2
    #TRAIN_TEST_SPLIT = 0.7
    #VALIDATION_SPLIT = 0.09
    #SONG_EMBEDDING_SIZE = 100
    #TAG_EMBEDDING_SIZE = 75
    #SKIP_ACTIONS = 2
    #SKIP_EMBEDDING_SIZE = 10
    #DROPOUT = 0.2
    #V_LAYER_SIZE = 50

    ABSOLUTE_PATH = ABSOLUTE_PATH + "{0}data{0}skips{0}".format(os.sep)
    # Load all data as dataframes.
    with open(ABSOLUTE_PATH + "sessions_10sessions_2plays_binaryskip_tf.json", "r") as source:
        sessions = pd.read_json(source, orient="index")

    with open(ABSOLUTE_PATH + "tracks_10sessions_2plays_binaryskip_tf.json", "r") as source:
        songs = pd.read_json(source, orient="index")
        songs.index = songs.index + 1

    with open(ABSOLUTE_PATH + "users_10sessions_2plays_binaryskip_tf.json", "r") as source:
        users = pd.read_json(source, orient="index")

    print("Created dataframes...")

    # Grab total number of tags.
    with open(ABSOLUTE_PATH + "tags_10sessions_2plays_binaryskip_tf.json", "r") as source:
        total_tags = len(json.load(source))

    # Create training and testing splits. 
    _, train_sessions, _, test_sessions = data_handling_tf.get_split(users, sessions, TRAIN_TEST_SPLIT, VALIDATION_SPLIT)
    # Create target columns
    train_sessions["target"] = train_sessions["tracks"].apply(lambda idxs: idxs[-1])
    test_sessions["target"] = test_sessions["tracks"].apply(lambda idxs: idxs[-1])
    train_sessions["target"] = train_sessions["target"] - 1
    test_sessions["target"] = test_sessions["target"] - 1
    # Remove target column track id and tag ids from input
    train_sessions["tracks"] = train_sessions["tracks"].apply(lambda idxs: idxs[:-1])
    train_sessions["tags"] = train_sessions["tags"].apply(lambda idxs: idxs[:-1])
    train_sessions["skips"] = train_sessions["skips"].apply(lambda idxs: idxs[:-1])
    test_sessions["tracks"] = test_sessions["tracks"].apply(lambda idxs: idxs[:-1])
    test_sessions["tags"] = test_sessions["tags"].apply(lambda idxs: idxs[:-1])
    test_sessions["skips"] = test_sessions["skips"].apply(lambda idxs: idxs[:-1])
    # Create Datasets for each set
    train_records = [tuple(row) for row in train_sessions.to_numpy()]
    test_records = [tuple(row) for row in test_sessions.to_numpy()]

    def train_generator():
        for record in train_records:
            yield record[:-1], record[-1]

    def test_generator():
        for record in test_records:
            yield record[:-1], record[-1]

    train_dataset = tf.data.Dataset.from_generator(train_generator, 
                    output_types=((tf.int32, tf.int32, tf.int32), tf.int32),
                    output_shapes=(((None,), (None, None,), (None,)), ()))
    train_batched = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=(([None], [None, None], [None]), []))
    train_batched = train_batched.prefetch(tf.data.experimental.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_generator(test_generator, 
                    output_types=((tf.int32, tf.int32, tf.int32), tf.int32),
                    output_shapes=(((None,), (None, None,), (None,)), ()))
    test_dataset = test_dataset.padded_batch(TEST_BATCH_SIZE, padded_shapes=(([None], [None, None], [None]), []))

    print("Created tf datasets...")

    # https://www.tensorflow.org/xla
    tf.config.optimizer.set_jit(True)

    #stabr = stabr_skip_lstm.STABR(
    stabr = stabr_skip_tf.STABR(len(songs) + 1,
                                SONG_EMBEDDING_SIZE,
                                total_tags + 1, 
                                TAG_EMBEDDING_SIZE,
                                SKIP_ACTIONS + 1, 
                                SKIP_EMBEDDING_SIZE, 
                                HIDDEN_SIZE, 
                                DROPOUT, 
                                V_LAYER_SIZE)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    stabr.compile(optimizer=optimizer, 
                  loss=loss_fn, 
                  metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        #"stabr_lstm_skips/stabr_{epoch}.ckpt",
        "models/stabr_skip_{epoch}.ckpt",
        save_weights_only=True,
        period=1
    )

    print("Created network...")

    if "train" in sys.argv:
        stabr.fit(train_batched, epochs=EPOCHS, verbose=1, callbacks=[checkpoint_cb], shuffle=False)
    elif "eval" in sys.argv:
        eval_model_path = ABSOLUTE_PATH + "models{0}stabr_skip_{1}.ckpt".format(os.sep, sys.argv[2])
        for k in [10, 20, 30, 40, 50]:
            stabr_eval = stabr_skip_tf.STABR(len(songs) + 1, 
                                             SONG_EMBEDDING_SIZE,
                                             total_tags + 1, 
                                             TAG_EMBEDDING_SIZE,
                                             SKIP_ACTIONS + 1, 
                                             SKIP_EMBEDDING_SIZE, 
                                             HIDDEN_SIZE, 
                                             DROPOUT, 
                                             V_LAYER_SIZE)
            stabr_eval.compile(optimizer=optimizer, 
                               loss=loss_fn, 
                               metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)])
            init_data = iter(train_batched).next()
            inp, target = init_data
            stabr_eval.train_on_batch(inp, target)
            stabr_eval.load_weights(eval_model_path)
            if sys.argv[3] == "hr":
                stabr_eval.evaluate(test_dataset)
            elif sys.argv[3] == "mrr":
                eval_mrr(stabr_eval, test_dataset, k)
    elif "checkpoint" in sys.argv:
        EPOCH = int(sys.argv[2])
        eval_model_path = "models{0}stabr_skip_{1}.ckpt".format(os.sep, EPOCH)
        init_data = iter(train_batched).next()
        inp, target = init_data
        stabr.train_on_batch(inp, target)
        stabr.load_weights(eval_model_path)
        stabr.fit(train_batched, epochs=EPOCHS, verbose=1, callbacks=[checkpoint_cb], shuffle=False, initial_epoch=EPOCH)
    else:
        print("Missing arguments.")
