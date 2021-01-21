import kerastuner as kt
import tensorflow as tf
import pandas as pd
import json
import os
from pathlib import Path
from STABR_tf import data_handling_tf, stabr_tf

os.environ["TF_XLA_FLAGS"] = ("--tf_xla_cpu_global_jit " + os.environ.get("TF_XLA_FLAGS", ""))

##### DATASET PREPARATION #####

ABSOLUTE_PATH = str(Path(__file__).resolve().parents[2].absolute())
TRAIN_TEST_SPLIT = 0.7
VALIDATION_SPLIT = 0.09

# Load all data as dataframes.
with open(ABSOLUTE_PATH + "{0}data{0}sessions_10sessions_2plays_tf.json".format(os.sep), "r") as source:
    sessions = pd.read_json(source, orient="index")

with open(ABSOLUTE_PATH + "{0}data{0}tracks_10sessions_2plays_tf.json".format(os.sep), "r") as source:
    songs = pd.read_json(source, orient="index")
    songs.index = songs.index + 1

with open(ABSOLUTE_PATH + "{0}data{0}users_10sessions_2plays_tf.json".format(os.sep), "r") as source:
    users = pd.read_json(source, orient="index")

print("Created dataframes...")

# Grab total number of tags.
with open(ABSOLUTE_PATH + "{0}data{0}tags_10sessions_2plays_tf.json".format(os.sep)) as source:
    total_tags = len(json.load(source))

# Create training and testing splits. 
_, train_sessions, val_sessions, _ = data_handling_tf.get_split(users, sessions, TRAIN_TEST_SPLIT, validation_split=VALIDATION_SPLIT)
# create target columns
train_sessions["target"] = train_sessions["tracks"].apply(lambda idxs: idxs[-1])
val_sessions["target"] = val_sessions["tracks"].apply(lambda idxs: idxs[-1])
train_sessions["target"] = train_sessions["target"] - 1
val_sessions["target"] = val_sessions["target"] - 1
#remove target column track id and tag ids from input
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
                output_types=((tf.int32, tf.int32), tf.int32),
                output_shapes=(((None,), (None, None,)), ()))
train_dataset = train_dataset.padded_batch(32, padded_shapes=(([None], [None, None]), []))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(val_generator, 
              output_types=((tf.int32, tf.int32), tf.int32),
              output_shapes=(((None,), (None, None,)), ()))
val_dataset = val_dataset.padded_batch(100, padded_shapes=(([None], [None, None]), []))

##### MODEL AND SEARCH SETUP #####

HIT_RATIO_K = 10

def build_model(hp):
    model = stabr_tf.STABR(
        len(songs) + 1,
        hp.Int("song_embedding_size", min_value=25, max_value=100, step=25, default=50),
        total_tags + 1,
        hp.Int("tag_embedding_size", min_value=50, max_value=100, step=25, default=50),
        hp.Int("hidden_size", min_value=32, max_value=128, step=32, default=50),
        hp.Choice("dropout", values=[0.1, 0.2, 0.3], default=0.1),
        hp.Int("v_layer_size", min_value=25, max_value=100, step=25, default=50))
    metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=HIT_RATIO_K)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=hp.Choice("learning_rate", values=[5e-2, 1e-1, 15e-2, 2e-1], default=1e-1))
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])

    return model

tuner = kt.tuners.Hyperband(
    build_model,
    objective="val_sparse_top_k_categorical_accuracy",
    max_epochs=75,
    hyperband_iterations=1,
    project_name="hs_stabr_"
)

earlystopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_sparse_top_k_categorical_accuracy", min_delta=0.005, patience=5, mode="max")

##### RUN SEARCH #####

tuner.search(train_dataset, validation_data=val_dataset, epochs=75, callbacks=[earlystopping_cb])
best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps)
best_model = tuner.get_best_models()[0]
best_model.save_weights("hyper_search_models/model", format="tf")
