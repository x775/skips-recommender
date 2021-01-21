import pandas as pd
import os
import sys
import json
from pathlib import Path
import tensorflow as tf
import stabr_custom_tf
import data_handling_tf
import kerastuner as kt
import numpy as np


if __name__ == "__main__":
    BATCH_SIZE = 32
    VAL_BATCH_SIZE = 100
    VALIDATION_SIZE = 0.1
    NP_RANDOM_SEED = 0

    ABSOLUTE_PATH = str(Path(__file__).resolve().parents[0].absolute())
    EVAL_MODEL_PATH = ""

    """ TAGS                    TRACKS
        training_0 54170        training_0 216054
        training_1 52441        training_1 210736
        training_2 50504        training_2 200469
        training_3 51153        training_3 207245
        training_4 52616        training_4 218665
        testing_0 7620          testing_0 16305
        testing_1 6898          testing_1 14026
        testing_2 6682          testing_2 14472
        testing_3 7108          testing_3 15146
        testing_4 8607          testing_4 20318
    """
    TRAINING_SET = "{0}{1}data{1}sessions_30music_training_0.json".format(ABSOLUTE_PATH, os.sep)
    TESTING_SET = "{0}{1}data{1}sessions_30music_testing_0.json".format(ABSOLUTE_PATH, os.sep)
    # Update number for each slice.
    TOTAL_SONGS = 216054
    TOTAL_TAGS = 54170

    tf.config.optimizer.set_jit(True)
    
    training_set = pd.read_json(TRAINING_SET, orient="index")

    # split training data into training and validation sets
    n_samples = len(training_set)
    sidx = np.arange(n_samples, dtype="int32")
    np.random.seed(NP_RANDOM_SEED)
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1 - VALIDATION_SIZE)))
    train_data = training_set.iloc[sidx[:n_train]]
    val_data = training_set.iloc[sidx[n_train:]]
    
    train_sessions = data_handling_tf.get_subsessions(train_data)
    val_sessions = data_handling_tf.get_subsessions(val_data)
    # Create target columns
    train_sessions["target"] = train_sessions["tracks"].apply(lambda idxs: int(idxs[-1]))
    train_sessions["target"] = train_sessions["target"] - 1
    val_sessions["target"] = val_sessions["tracks"].apply(lambda idxs: int(idxs[-1]))
    val_sessions["target"] = val_sessions["target"] - 1

    # Remove target column track id and tag ids from input
    train_sessions["tracks"] = train_sessions["tracks"].apply(lambda idxs: idxs[:-1])
    train_sessions["tags"] = train_sessions["tags"].apply(lambda idxs: idxs[:-1])
    val_sessions["tracks"] = val_sessions["tracks"].apply(lambda idxs: idxs[:-1])
    val_sessions["tags"] = val_sessions["tags"].apply(lambda idxs: idxs[:-1])

    # Create Datasets for each set
    train_records = [tuple(row) for row in train_sessions.to_numpy()]
    val_records = [tuple(row) for row in val_sessions.to_numpy()]

    def train_generator():
        for record in train_records:
            yield record[:-1], record[-1]

    def val_generator():
        for record in val_records:
            yield record[:-1], record[-1]

    train_dataset = tf.data.Dataset.from_generator(train_generator, 
                    output_types=((tf.int32, tf.int32), tf.int32), output_shapes=(((None,), (None, None,)), ()))
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=(([None], [None, None]), []))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(val_generator, 
                    output_types=((tf.int32, tf.int32), tf.int32), output_shapes=(((None,), (None, None,)), ()))
    val_dataset = val_dataset.padded_batch(VAL_BATCH_SIZE, padded_shapes=(([None], [None, None]), []))

    print("Created TF datasets...")
    
    # Update value depending on run.
    HIT_RATIO_K = 10

    def build_model(hp):
        model = stabr_custom_tf.STABR(
            TOTAL_SONGS + 1,
            hp.Int("song_embedding_size", min_value=25, max_value=100, step=25, default=50),
            TOTAL_TAGS + 1,
            hp.Int("tag_embedding_size", min_value=50, max_value=100, step=25, default=50),
            hp.Int("hidden_size", min_value=32, max_value=96, step=32, default=32),
            hp.Choice("dropout", values=[0.1, 0.2], default=0.1),
            hp.Int("v_layer_size", min_value=25, max_value=75, step=25, default=50))
        metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=HIT_RATIO_K)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=hp.Choice("learning_rate", values=[5e-2, 1e-1, 15e-2], default=1e-1))
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])

        return model

    tuner = kt.tuners.Hyperband(
        build_model,
        objective="val_sparse_top_k_categorical_accuracy",
        max_epochs=20,
        hyperband_iterations=1,
        project_name="stabr_30music"
    )

    earlystopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_sparse_top_k_categorical_accuracy", min_delta=0.005, patience=5, mode="max")

    ##### RUN SEARCH #####
    tuner.search(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[earlystopping_cb])
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps)
    best_model = tuner.get_best_models()[0]
    best_model.save_weights("hyper_search_stabr_30music/model", format="tf")