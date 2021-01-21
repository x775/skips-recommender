import os
import sys
import json
import math
import numpy as np
import pandas as pd
import data_handling_skip
import multiprocessing as mp

from queue import Queue
from scipy import sparse
from pathlib import Path
from threading import Thread
from timeit import default_timer as timer
from sklearn.metrics.pairwise import cosine_similarity


def train_sknn(train_sessions, track_count):
    sessions = train_sessions.tracks.values.tolist()
    sessions_skips = train_sessions.skips.values.tolist()
    sknn_mat = sparse.lil_matrix((len(sessions), track_count), dtype=np.int16)

    for i, session in enumerate(sessions):
        unique_tracks = list(set(session))
        track_plays = dict(zip(unique_tracks, [0] * len(unique_tracks)))
        skips = sessions_skips[i]
        for j, track in enumerate(session):
            if skips[j] == 1:
                track_plays[track] += 1
        for key in track_plays:
            sknn_mat[i, key - 1] = track_plays[key]

    return sknn_mat.tocsr()

trained_sknn = None 
test_sessions_list = None
test_skips_list = None
results_queue = mp.Queue()

def generate_batch(sessions, batch_size):
    for i in range(0, len(sessions), batch_size):
        yield sessions[i:i + batch_size]
        

def eval_sknn(trained_sknn, start_idx, end_idx, num_of_tracks, nn=10, topk=10, batch_size=1000):
    total = 0
    correct = 0
    cumulative_rank = 0
    batches = []
    
    for i in range(start_idx, end_idx, batch_size):
        batches.append(list(zip(test_sessions_list[i:i + batch_size], test_skips_list[i:i + batch_size])))

    for batch in batches:
        target_dict = {}
        test_matrix = sparse.lil_matrix((len(batch), num_of_tracks), dtype=np.int16)
    
        for i, session in enumerate(batch):
            input_tracks = session[0][:-1]
            target = session[0][-1] - 1
            target_dict[i] = target
            skips = session[1]

            unique_tracks = list(set(input_tracks))
            track_plays = dict(zip(unique_tracks, [0] * len(unique_tracks)))
            for j, track in enumerate(input_tracks):
                if skips[j] == 1:
                    track_plays[track] += 1
            for key in track_plays:
                test_matrix[i, key - 1] = track_plays[key]

        test_matrix = test_matrix.tocsr()
        similarity_matrix = cosine_similarity(test_matrix, trained_sknn, dense_output=False)
        test_mean_ratings = test_matrix.mean(axis=1)

        print("Calculating scores...")

        for i in range(len(batch)):
            neighbours = np.argpartition(similarity_matrix[i].toarray()[0], -nn)[-nn:]
            rating_denom = similarity_matrix[i, neighbours].sum()
            test_mean_rating = test_mean_ratings[i].item()
            train_mean_ratings = trained_sknn[neighbours].mean(axis=1)
            train_simililarities = similarity_matrix[i, neighbours]
            neighbour_ratings = trained_sknn[neighbours]
            track_scores = np.multiply(neighbour_ratings - train_mean_ratings,
                                       train_simililarities.toarray().squeeze(0)[:, np.newaxis]).sum(axis=0)
            track_ratings = track_scores / rating_denom if rating_denom != 0 else track_scores
            track_scores = test_mean_rating + track_ratings
            topk_tracks = np.argsort(np.asarray(track_scores).reshape(-1))[-topk:]
            topk_tracks = np.flip(topk_tracks)
        
            total += 1
            if target_dict[i] in topk_tracks:
                correct += 1
                rank = np.where(topk_tracks == target_dict[i])[0][0]
                cumulative_rank += 1 / (rank + 1)

            if i % 10 == 9:
                print("PID-{}\t{}/{}\tnn:{}, k:{}".format(os.getpid(), correct, total, nn, topk))

    results_queue.put((correct, total, cumulative_rank))



if __name__ == "__main__":
    NUM_OF_CORES = 32
    TRAIN_TEST_SPLIT = 0.7
    VALIDATION_SPLIT = 0.09

    if "allskips" in sys.argv:
        # Load all data as dataframes.
        ABSOLUTE_PATH = str(Path(__file__).resolve().parents[1].absolute()) + "{0}data{0}all_sessions_skips{0}".format(os.sep)
        with open(ABSOLUTE_PATH + "sessions_10sessions_2plays_allskips_tf.json", "r") as source:
            sessions = pd.read_json(source, orient="index")

        with open(ABSOLUTE_PATH + "tracks_10sessions_2plays_allskips_tf.json", "r") as source:
            tracks = pd.read_json(source, orient="index")
            NUM_OF_TRACKS = len(tracks)

        with open(ABSOLUTE_PATH + "users_10sessions_2plays_allskips_tf.json", "r") as source:
            users = pd.read_json(source, orient="index")
    else:
        # Load all data as dataframes.
        ABSOLUTE_PATH = str(Path(__file__).resolve().parents[1].absolute()) + "{0}data{0}skips{0}".format(os.sep)
        with open(ABSOLUTE_PATH + "sessions_10sessions_2plays_binaryskip_tf.json", "r") as source:
            sessions = pd.read_json(source, orient="index")

        with open(ABSOLUTE_PATH + "tracks_10sessions_2plays_binaryskip_tf.json", "r") as source:
            tracks = pd.read_json(source, orient="index")
            NUM_OF_TRACKS = len(tracks)

        with open(ABSOLUTE_PATH + "users_10sessions_2plays_binaryskip_tf.json", "r") as source:
            users = pd.read_json(source, orient="index")


    print("Created dataframes...")

    # Create train and testing split. 
    _, train_sessions, _, test_sessions = data_handling_skip.get_split(users, sessions, TRAIN_TEST_SPLIT, VALIDATION_SPLIT)
    #train_sessions.drop("tags", axis=1, inplace=True)
    test_sessions.drop("tags", axis=1, inplace=True)
    test_sessions_list = test_sessions.tracks.values.tolist()
    test_skips_list = test_sessions.skips.values.tolist()

    if sys.argv[1] == "train":
        trained_sknn = train_sknn(train_sessions, NUM_OF_TRACKS)
        sparse.save_npz("trained_sknn_skip_allskips_tf.npz", trained_sknn)
    if sys.argv[1] == "eval":
        # Load the saved training sparse matrix; in csr format.
        trained_sknn = sparse.load_npz("trained_sknn_skip_tf.npz")

    TOTAL = len(test_sessions_list)
    SESSIONS_PER_CORE = math.ceil(TOTAL / NUM_OF_CORES)
    
    neighbours = [10, 25, 50, 75, 100, 125, 150]
    topk = [1, 5, 10, 20, 30, 40, 50]
    for neighbour in neighbours:
        for k in topk:
            processes = []
            for i in range(NUM_OF_CORES):
                process_start = i * SESSIONS_PER_CORE
                process_end = process_start + SESSIONS_PER_CORE
                p = mp.Process(target=eval_sknn, args=(trained_sknn, process_start, process_end, NUM_OF_TRACKS, neighbour, k))
                processes.append(p)
                p.start()
                print("Started process", i)
        
            for p in processes:
                p.join()
                
            results = [results_queue.get() for p in processes]
            correct = sum([pair[0] for pair in results])
            total = sum([pair[1] for pair in results])
            cumulative_rank = sum([pair[2] for pair in results])

            hit_ratio = correct / total
            print("Hit ratio @ {}: {} ({} hits in {} samples)".format(
                k, hit_ratio, correct, total))

            mrr = cumulative_rank / total
            print("Mean Reciprocal Rank @ {}: {}".format(k, mrr))

            with open("sknn_skip_150_results.txt", "a") as out:
                out.write("Hit ratio @ {}: {} ({} hits in {} samples): {} neighbours\n"\
                        .format(k, hit_ratio, correct, total, neighbour))
                out.write("Mean Reciprocal Rank @ {}: {}\n".format(k, mrr))
