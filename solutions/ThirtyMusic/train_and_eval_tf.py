import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from solutions.ThirtyMusic import stabr_tf
from solutions.ThirtyMusic import data_handling_tf


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


def eval_hit_ratio_at_k(network, test_data, k=10):
    correct = 0
    total = 0
    for step, data in enumerate(test_data):
        inp, targets = data
        logits = network(inp, training=False)
        log_probs = tf.nn.log_softmax(logits, axis=1)

        for i in range(targets.shape[0]):
            _, output_idxs = tf.nn.top_k(log_probs[i], k=k)
            total += 1
            if targets[i] in output_idxs:
                correct += 1
        
        if step % 100 == 99:
            print("{}/{}".format(correct, total))
    
    hit_ratio = correct / total
    print("Hit ratio @ {}: {} ({} hits in {} samples)".format(k, hit_ratio, correct, total))


class LossCallback(tf.keras.callbacks.Callback):
        def __init__(self, print_after=100):
            super(LossCallback, self).__init__()
            self.print_after = print_after
        
        def on_train_batch_end(self, batch, logs=None):
            if batch % self.print_after == self.print_after - 1:
                print("Batch {}, loss:{:7.3f}".format(batch + 1, logs["loss"]))


def run_train_test():
    # Manually define relevant hyperparameters.
    EPOCHS = 100
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 50
    VAL_BATCH_SIZE = 100
    HIDDEN_SIZE = 64
    LEARNING_RATE = 0.15
    TRAIN_TEST_SPLIT = 0.7
    SONG_EMBEDDING_SIZE = 75
    TAG_EMBEDDING_SIZE = 100
    NP_RANDOM_SEED = 0
    VALIDATION_SIZE = 0.1
    DROPOUT = 0.1
    V_LAYER_SIZE = 25
    
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
        TRAINING_SET = "{0}{1}data{1}sessions_30music_training_{2}.json".format(ABSOLUTE_PATH, os.sep, index)
        TESTING_SET = "{0}{1}data{1}sessions_30music_testing_{2}.json".format(ABSOLUTE_PATH, os.sep, index)
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

        train_sessions = data_handling_tf.get_subsessions(train_data)
        val_sessions = data_handling_tf.get_subsessions(val_data)
        test_sessions = data_handling_tf.get_subsessions(pd.read_json(TESTING_SET, orient="index"))

        # Create target columns
        train_sessions["target"] = train_sessions["tracks"].apply(lambda idxs: int(idxs[-1]))
        test_sessions["target"] = test_sessions["tracks"].apply(lambda idxs: int(idxs[-1]))
        val_sessions["target"] = val_sessions["tracks"].apply(lambda idxs: int(idxs[-1]))
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

        def train_generator():
            for record in train_records:
                yield record[:-1], record[-1]

        def val_generator():
            for record in val_records:
                yield record[:-1], record[-1]

        def test_generator():
            for record in test_records:
                yield record[:-1], record[-1]

        train_dataset = tf.data.Dataset.from_generator(train_generator, 
                        output_types=((tf.int32, tf.int32), tf.int32), output_shapes=(((None,), (None, None,)), ()))
        train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=(([None], [None, None]), []))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(val_generator, 
                    output_types=((tf.int32, tf.int32), tf.int32), output_shapes=(((None,), (None, None,)), ()))
        val_dataset = val_dataset.padded_batch(VAL_BATCH_SIZE, padded_shapes=(([None], [None, None]), []))

        test_dataset = tf.data.Dataset.from_generator(test_generator, 
                        output_types=((tf.int32, tf.int32), tf.int32), output_shapes=(((None,), (None, None,)), ()))
        test_dataset = test_dataset.padded_batch(TEST_BATCH_SIZE, padded_shapes=(([None], [None, None]), []))

        print("Created TF datasets...")

        stabr = stabr_tf.STABR(TOTAL_SONGS + 1, 
                               SONG_EMBEDDING_SIZE, 
                               TOTAL_TAGS + 1, 
                               TAG_EMBEDDING_SIZE, 
                               HIDDEN_SIZE,
                               DROPOUT, 
                               V_LAYER_SIZE)
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            "stabr_30music_models/stabr_30music_set" + str(index) + "_{epoch}.ckpt",
            save_weights_only=True,
            period=1
        )

        print("Created network...")

        if "train" in sys.argv:
            stabr.compile(optimizer=optimizer, 
                          loss=loss_fn, 
                          metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])
            stabr.fit(train_dataset, 
                      epochs=EPOCHS, 
                      verbose=1, 
                      callbacks=[checkpoint_cb],
                      validation_data=val_dataset,
                      shuffle=False)
        elif "eval" in sys.argv:
            eval_model_path = "stabr_30music_models/stabr_30music_set{}_{}.ckpt".format(index, sys.argv[2])
            for k in [1, 5, 10, 20, 30, 40, 50]:
                stabr.compile(optimizer=optimizer, 
                              loss=loss_fn, 
                              metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])
                init_data = iter(train_dataset).next()
                inp, target = init_data
                stabr.train_on_batch(inp, target)
                stabr.load_weights(eval_model_path)
                if sys.argv[3] == "hr":
                    stabr.evaluate(test_dataset)
                elif sys.argv[3] == "mrr":
                    eval_mrr(stabr, test_dataset, k)
        elif "checkpoint" in sys.argv:
            EPOCH = sys.argv[2]
            eval_model_path = "stabr_30music_models/stabr_30music_set{}_{}.ckpt".format(index, sys.argv[2])
            stabr.compile(optimizer=optimizer, 
                          loss=loss_fn, 
                          metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])
            init_data = iter(train_dataset).next()
            inp, target = init_data
            stabr.train_on_batch(inp, target)
            stabr.load_weights(eval_model_path)
            stabr.fit(train_dataset, 
                      epochs=EPOCHS, 
                      verbose=1, 
                      callbacks=[checkpoint_cb],
                      validation_data=val_dataset, 
                      shuffle=False, 
                      initial_epoch=EPOCH)
        else:
            print("Missing arguments.")
