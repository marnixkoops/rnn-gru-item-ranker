import numpy as np
import pandas as pd
import time
import datetime
import warnings
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

import mlflow
from ml_metrics import average_precision
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("notebook")
warnings.simplefilter(action="ignore", category=FutureWarning)

####################################################################################################
# 🚀 EXPERIMENT SETTINGS
####################################################################################################

# run settings
DRY_RUN = True  # runs flow on small subset of data for speed and disables mlfow tracking
LOGGING = False  # mlflow experiment logging
WEEKS_OF_DATA = 1  # use 1, 2, 3 or 4 weeks worth of data

# define where we run and on which device (GPU/CPU)
# GPU_AVAILABLE = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
all_devices = str(device_lib.list_local_devices())
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if "Tesla P100" in all_devices:
    DEVICE = "Tesla P100 GPU"
    MACHINE = "cloud"
    # tf.config.experimental.set_memory_growth(gpu_devices[0], True)  # no allocating memory upfront
elif "GPU" in all_devices:
    DEVICE = "GPU"
    MACHINE = "cloud"
    # tf.config.experimental.set_memory_growth(gpu_devices[0], True)
elif "CPU" in all_devices:
    DEVICE = "CPU"
    MACHINE = "local"

print("🧠 Running TensorFlow version {} on {}".format(tf.__version__, DEVICE))

# data constants
N_TOP_ITEMS = None  # we have ~1600 product_types currently
MIN_ITEMS_TRAIN = 1  # sequences with less products (excluding target) are invalid and removed
MIN_ITEMS_TEST = 1  # sequences with less products (excluding target) are invalid and removed
WINDOW_LEN = 6  # fixed moving window size for generating input-sequence/target rows for training
PRED_LOOKBACK = 6  # number of most recent products used per sequence in the test set to predict on
TOP_K_OUTPUT_LEN = 12  # number of top K item probabilities to extract from the probabilities

# model constants
EMBED_DIM = 24  # number of dimensions for the embeddings
N_HIDDEN_UNITS = 48  # number of units in the GRU layers
MAX_EPOCHS = 12  # maximum number of epochs to train for
BATCH_SIZE = 512  # batch size for training
DROPOUT = 0.2  # node dropout
RECURRENT_DROPOUT = 0.2  # recurrent state dropout during training, fast CuDNN GPU requires 0!
LEARNING_RATE = 0.004
OPTIMIZER = tf.keras.optimizers.Nadam(
    learning_rate=LEARNING_RATE
)  # note, tested a couple (RMSProp, Adam, Nadam), Adam and Nadam both seem fast with good results

# Automatic FP16 mixed-precision training instead of FP32 for gradients and model weights
# See: https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow-amp
# Needs more investigation in terms of speed, gives a warning for memory heavy tensor conversion
# OPTIMIZER = tf.train.experimental.enable_mixed_precision_graph_rewrite(OPTIMIZER)

# training constants
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1  # note, creates a gap in time between train/test, no val improves performance
TEST_RATIO = 0.1  # note, this % results in more samples when more weeks of data are used
SHUFFLE_TRAIN_SET = True  # shuffles the training sequences (row-wise), seems smart for training
SHUFFLE_TRAIN_AND_VAL_SET = False  # shuffles both the training and validation sequences
DATA_IMBALANCE_CORRECTION = False  # Supply product weights during model training to avoid bias
FILTER_REPEATED_UNIQUES = False  # Filters sequences with only 1 unique item e.g. [1, 1, 1, 1, 1]

# dry run constants for development and debugging
if DRY_RUN:
    SEQUENCES = 10000
    N_TOP_ITEMS = None
    EMBED_DIM = 16
    N_HIDDEN_UNITS = 32
    BATCH_SIZE = 32
    MAX_EPOCHS = 5
    LEARNING_RATE = 0.01

####################################################################################################
# 🚀 GOOGLE CLOUD INPUT DATA
####################################################################################################


def print_memory_footprint(array):
    """Prints a statement with the memory size of the input array"""
    print("     Memory footprint of training array: {:.4} GigaBytes".format(array.nbytes * 1e-9))


def downcast_numeric_datatypes(df):
    float_cols = df.select_dtypes(include=["float"])
    int_cols = df.select_dtypes(include=["int"])

    for cols in float_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast="float")
    for cols in int_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast="integer")

    return df


def query_product_sequence_data():
    print("     Downloading sequence data from Google BigQuery")
    query = """
SELECT
  coolblue_cookie_id,
  visit_date,
  product_sequence
FROM
  `coolblue-bi-data-science-exp.recommender_systems.ga_product_sequence`
WHERE
  _PARTITIONTIME IN (
  SELECT
    MAX(_PARTITIONTIME)
  FROM
    `coolblue-bi-data-science-exp.recommender_systems.ga_product_sequence`)
"""
    return pd.read_gbq(query, "coolblue-bi-data-science-exp", dialect="standard")


def query_product_map_data():
    print("     Downloading product mapping data from Google BigQuery")
    query = """
SELECT
  product_id,
  product_type_id,
  product_type_name,
FROM
  `coolblue-bi-platform-prod.product.products`
"""
    return pd.read_gbq(query, "coolblue-bi-platform-prod", dialect="standard")


# download data
sequence_df = query_product_sequence_data()
product_df = query_product_map_data()
product_df = downcast_numeric_datatypes(product_df)

sequence_df_len = len(sequence_df)
sequence_df = sequence_df.drop_duplicates(keep="first")  # also checks for visit_date + id
MIN_DATE, MAX_DATE = sequence_df["visit_date"].min(), sequence_df["visit_date"].max()

print("     Dropped {} duplicate rows".format(sequence_df_len - len(sequence_df)))
print("     Data contains {} sequences from {} to {}".format(len(sequence_df), MIN_DATE, MAX_DATE))

####################################################################################################
# 🚀 PREPARE DATA FOR MODELING
####################################################################################################


def filter_valid_sequences(array, min_items=MIN_ITEMS_TRAIN):
    """Filters sequences that are not valid. Invalid sequences do not contain enough products or end
    in a 0 (padding) which can occur due to creating subsequences of longer sequences for training.
    Args:
        array (array): Input matrix with sequences.
        min_items (int): Treshold for filtering invalid sequences, +1 due to excluding target.
    Returns:
        array: Valid sequences
    """
    min_items = min_items + 1  # correction since we need a target for training/testing!
    pre_len = len(array)
    array = array[array[:, -1] != 0]  # ending in 0 is a duplicate subsequence or empty sequence
    min_product_mask = np.sum((array != 0), axis=1) >= min_items  # create mask for min products
    valid_sequences = array[min_product_mask].copy()
    print("     Removed {} invalid sequences".format(pre_len - len(valid_sequences)))
    print("     Kept {} valid sequences".format(len(valid_sequences)))
    return valid_sequences


# generate subsequences from longer sequences with a moving window for model training
# this numpy function is fast but a bit tricky, be sure to validate output when changing stuff
def generate_train_test_pairs(array, input_length=WINDOW_LEN, step_size=1):
    """Creates multiple subsequences out of longer sequences in the matrix to be used for training.
    Output shape is based on the input_length. Note that the output width is input_length + 1 since
    we later take the last column of the matrix to obtain an input matrix of width input_length and
    a vector with corresponding targets (next item in that sequence).
    Args:
        array (array): Input matrix with equal length padded sequences.
        input_length (int): Size of sliding window, equal to desired length of input sequence.
        step_size (int): Can be used to skip items in the sequence.
    Returns:
        array: Reshaped matrix with # columns equal to input_length + 1 (input + target item).
    """
    shape = (array.size - input_length + 2, input_length + 1)
    strides = array.strides * 2
    window = np.lib.stride_tricks.as_strided(array, strides=strides, shape=shape)[0::step_size]
    return window.copy()


def filter_repeated_unique_sequences(array, min_items=1):
    """Remove (sub)sequences that only contain a repeated unique product (no info/noise?).
    Note that padding=0 is included as item but is actually not a unique view, hence the > min_items + 1.
    Args:
        array (type): Description of parameter `array`.
    Returns:
        (type): Description of returned object.
    """
    pre_len = len(array)
    unique_count_per_row = np.apply_along_axis(func1d=lambda x: len(set(x)), arr=array, axis=1)
    repeated_uniques_mask = (
        unique_count_per_row > min_items + 1
    )  # +1 since padding is included as 'unique' item
    valid_sequences = array[repeated_uniques_mask].copy()
    print(
        "     Removed {} sequences containing a single repeated unique item".format(
            pre_len - len(valid_sequences)
        )
    )
    print("     Kept {} valid sequences".format(len(valid_sequences)))
    return valid_sequences


t_prep = time.time()  # start timer for preparing data

print("\n💾 Processing data")
print("     Tokenizing, padding, filtering & splitting sequences")
# define product_to_pt_tokenizer to encode sequences and include N most popular items (occurence)
product_to_pt_map_dict = dict(
    zip(product_df["product_id"].astype(str), product_df["product_type_id"].astype(str))
)
product_to_pt_tokenizer = keras.preprocessing.text.Tokenizer()
product_to_pt_tokenizer.word_index = product_to_pt_map_dict
# product_to_pt_tokenizer.fit_on_texts(sequence_df["product_sequence"])  # encode string sequences
sequences = product_to_pt_tokenizer.texts_to_sequences(sequence_df["product_sequence"])
del sequence_df
gc.collect()

# tokenize product_types to encode sequences from 1 to N
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)

# pre-pad sequences with 0's, length is based on longest present sequence
# this is required to transform the variable length sequences into equal train/test pairs
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="pre")
print_memory_footprint(padded_sequences)
del sequences
gc.collect()

# split into train/test subsets before reshaping sequence for training/validation
# (since we subsample longer sequences of a single customer into multiple train/validation pairs,
# while we do not wish to predict on multiple sequences of a single customer
test_index = int(TEST_RATIO * len(padded_sequences))
padded_sequences_train = padded_sequences[:-test_index].copy()
padded_sequences_test = padded_sequences[-test_index:].copy()

padded_sequences_train = filter_valid_sequences(padded_sequences_train, min_items=MIN_ITEMS_TRAIN)
padded_sequences_test = filter_valid_sequences(padded_sequences_test, min_items=MIN_ITEMS_TEST)

# determine most popular items, to be used in evaluation as baseline
unique, counts = np.unique(padded_sequences_train, return_counts=True)
frequency_dict = dict(zip(unique, counts))
pop_products = list(sorted(frequency_dict, key=frequency_dict.get, reverse=True))[1:11]

# how many items there are for the input layer
n_included_pt = len(tokenizer.word_index.values())
N_TOP_ITEMS = n_included_pt + 1

print("\n     Unique items present in the data: {}".format(N_TOP_ITEMS))
print("     Training & evaluating model on {} sequences".format(len(padded_sequences_train)))
print("     Testing recommendations on {} sequences".format(len(padded_sequences_test)))
print_memory_footprint(padded_sequences_train)

# generate a dictionary with unique value counts to be used as class_weight in model fit
# the idea is to combat against bias due to highly imbalanced training data
if DATA_IMBALANCE_CORRECTION:
    print("     Generating product occurence dictionary to be used as class weights against bias")
    product_token, product_count = np.unique(padded_sequences_train, return_counts=True)
    product_count_dict = dict(zip(product_token, product_count))
    del product_count_dict[0]  # drop 0 token, which is padding and not a product
else:
    product_count_dict = None

# clean up memory
del padded_sequences
gc.collect()


# generate sliding window of sequences with x=WINDOW_LEN input products and y=1 target product
print("\n     Reshaping into train/test subsequences with fixed window size for training")
padded_sequences_train = np.apply_along_axis(
    func1d=generate_train_test_pairs, axis=1, arr=padded_sequences_train
)
padded_sequences_train = np.vstack(padded_sequences_train).copy()  # stack sequences
print("     Generated {} subsequences for training/validation".format(len(padded_sequences_train)))

# filter sequences, note that due to reshaping invalid sequences can be re-introduced
padded_sequences_train = filter_valid_sequences(padded_sequences_train, min_items=MIN_ITEMS_TRAIN)

# HOW TO DEAL WITH THESE?
# TODO TODO TODO TODO TODO TODO TODO
if FILTER_REPEATED_UNIQUES:
    padded_sequences_train = filter_repeated_unique_sequences(padded_sequences_train)


print_memory_footprint(padded_sequences_train)


# shuffle training and validation sequences randomly (across rows, not within sequences ofcourse)
if SHUFFLE_TRAIN_AND_VAL_SET:
    padded_sequences_train = shuffle(padded_sequences_train)

# split sequences into subsets for training/validation/testing
# the last column of each row is the target product for each input subsequence
val_index = int(VAL_RATIO * len(padded_sequences_train))
X_train, y_train = padded_sequences_train[:-val_index, :-1], padded_sequences_train[:-val_index, -1]
X_val, y_val = padded_sequences_train[:val_index, :-1], padded_sequences_train[:val_index, -1]
X_test, y_test = padded_sequences_test[:, -(PRED_LOOKBACK + 1) : -1], padded_sequences_test[:, -1]

# shuffle training sequences randomly (across rows, not within sequences ofcourse)
if SHUFFLE_TRAIN_SET:
    X_train, y_train = shuffle(X_train, y_train)

# only train/test split (no validation), untested but this might improve MAP due to having no timegap
# X_train, y_train = padded_sequences_train[:, :-1], padded_sequences_train[:, -1]
# X_test, y_test = padded_sequences_test[:, -5:-1], padded_sequences_test[:, -1]

print("\n     Dropping some remainder rows to fit data into batches of {}".format(BATCH_SIZE))
train_index = len(X_train) - len(X_train) % BATCH_SIZE
val_index = len(X_val) - len(X_val) % BATCH_SIZE
test_index = len(X_test) - len(X_test) % BATCH_SIZE
X_train, y_train = X_train[:train_index, :], y_train[:train_index]
X_val, y_val = X_val[:val_index:, :], y_val[:val_index]
X_test, y_test = X_test[:test_index, :], y_test[:test_index]

print("     Final dataset dimensions:")
print("     Training X {}, y {}".format(X_train.shape, y_train.shape))
print("     Validation X {}, y {}".format(X_val.shape, y_val.shape))
print("     Testing X {}, y {}".format(X_test.shape, y_test.shape))

print("⏱️ Elapsed time for processing input data: {:.4} seconds".format(time.time() - t_prep))

del padded_sequences_train, padded_sequences_test
gc.collect()


####################################################################################################
# 🚀 DEFINE AND TRAIN RECURRENT NEURAL NETWORK
####################################################################################################

if LOGGING and not DRY_RUN:
    mlflow.start_run()  # start mlflow run for experiment tracking
t_train = time.time()  # start timer for training

print("\n🧠 Defining network")
tf.keras.backend.clear_session()  # clear potentially remaining network graphs in the memory
gc.collect()


def embedding_GRU_model(
    vocab_size=N_TOP_ITEMS,
    embed_dim=EMBED_DIM,
    num_units=N_HIDDEN_UNITS,
    batch_size=BATCH_SIZE,
    dropout=DROPOUT,
    recurrent_dropout=RECURRENT_DROPOUT,
):
    """Defines a RNN model with a trainable embedding input layer and GRU units.
    Args:
        vocab_size (int): Number of unique items included in the data.
        embed_dim (int): Number of embedding dimensions.
        num_units (int): Number of units for the GRU layer
        batch_size (int): Number of subsequences used in a single pass during training.
        dropout (float): Probability of dropping a node.
        recurrent_dropout (float): Probability of dropping a hidden state during training.

    Returns:
        tensorflow.keras.Sequential: Model object

    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                input_dim=N_TOP_ITEMS,
                output_dim=EMBED_DIM,
                batch_input_shape=[BATCH_SIZE, None],
                mask_zero=True,
            ),
            tf.keras.layers.GRU(
                units=N_HIDDEN_UNITS,
                activation="tanh",  # 'tanh' is required for CuDNN GPU support
                recurrent_activation="sigmoid",  # 'sigmoid' is required for CuDNN GPU support
                dropout=DROPOUT,
                recurrent_dropout=RECURRENT_DROPOUT,  # 0 is required for CuDNN GPU support
                return_sequences=False,
                unroll=False,
                use_bias=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
                reset_after=True,  # True is required for CuDNN GPU support
            ),
            tf.keras.layers.Dense(N_TOP_ITEMS, activation="sigmoid"),
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
    return model


model = embedding_GRU_model(
    vocab_size=N_TOP_ITEMS, embed_dim=EMBED_DIM, num_units=N_HIDDEN_UNITS, batch_size=BATCH_SIZE
)

# network info
print("\n     Network summary: \n{}".format(model.summary()))
total_params = model.count_params()
# model.get_config() # detailed model parameter settings

# early stopping monitor, stops training if no improvement in validation set for 1 epochs
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=1, verbose=1, restore_best_weights=True
)

print("     Training for a maximum of {} Epochs with batch size {}".format(MAX_EPOCHS, BATCH_SIZE))
model_history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    callbacks=[early_stopping_monitor],
    class_weight=product_count_dict,  # this is a dictionary with item occurence count
)

train_time = time.time() - t_train
print(
    "⏱️ Elapsed time for training network with {} parameters on {} sequences: {:.3} minutes".format(
        total_params, len(y_train), train_time / 60
    )
)

del X_val, y_val, y_train
gc.collect()


####################################################################################################
# 🚀 CREATE PREDICTIONS
####################################################################################################

print("\n🧠 Evaluating predictions of network")
t_pred = time.time()  # start timer for predictions
print("     Creating predictions on test set")


def generate_predicted_sequences(y_pred_probs, output_length=TOP_K_OUTPUT_LEN):
    """Function to extract predicted output sequences. Output is based on the predicted logit values
    where the highest probability corresponds to the first ranked item and so forth.
    Output positions are based on probability from high to low so the output sequence is ordered.
    To be used for obtaining multiple product ranked and calculating MAP@K values.
    Args:
        y_pred_probs (array): Predicted probabilities for all included items.
        output_length (int): Number of top K items to extract from the prediction array.
    Returns:
        array: Predictions matrix with shape [X_test, output_length]
    """
    # obtain indices of highest logit values, the position corresponds to the encoded item
    ind_of_max_logits = np.argpartition(y_pred_probs, -output_length)[-output_length:]
    # order the sequence, sorting the negative values ascending equals sorting descending
    ordered_predicted_sequences = ind_of_max_logits[np.argsort(-y_pred_probs[ind_of_max_logits])]

    return ordered_predicted_sequences


# model.predict occasionaly leads to memory leaking issues when input data is large.
# issue is known and should be fixed soon: https://github.com/keras-team/keras/issues/13118 and
# https://github.com/tensorflow/tensorflow/issues/33009
# y_pred_probs = model.predict(X_test, batch_size=BATCH_SIZE)

# predict in multiple batches to fit in GPU memory (array is 4 bytes * sequences * items = big)
dividing_row = len(X_test) // 3
remainder_due_to_batch_size = dividing_row % BATCH_SIZE  # needs to fit into BATCH_SIZE
dividing_row = dividing_row - remainder_due_to_batch_size

predicted_sequences = np.empty(
    [len(X_test), TOP_K_OUTPUT_LEN], dtype=np.int32
)  # pre-allocate required memory for array for efficiency

# first batch of recomendations
y_pred_probs = model.predict(X_test[:dividing_row])
predicted_sequences[:dividing_row] = np.apply_along_axis(
    func1d=generate_predicted_sequences,
    axis=1,
    arr=y_pred_probs,  # extract TOP_K_OUTPUT_LEN probabilities
)
del y_pred_probs
gc.collect()

# second batch of recomendations
y_pred_probs = model.predict(X_test[dividing_row : dividing_row * 2])
predicted_sequences[dividing_row : dividing_row * 2] = np.apply_along_axis(
    func1d=generate_predicted_sequences,
    axis=1,
    arr=y_pred_probs,  # extract TOP_K_OUTPUT_LEN probabilities
)
del y_pred_probs
gc.collect()

# third batch of recomendations
y_pred_probs = model.predict(X_test[dividing_row * 2 :])
predicted_sequences[dividing_row * 2 :] = np.apply_along_axis(
    func1d=generate_predicted_sequences,
    axis=1,
    arr=y_pred_probs,  # extract TOP_K_OUTPUT_LEN probabilities
)
# predicted_sequences = predicted_sequences.astype(np.int32)
del y_pred_probs
gc.collect()

# batched prediction loop to avoid memory leak issues in the model.predict call
# y_pred_probs = np.empty(
#     [len(X_test), N_TOP_ITEMS], dtype=np.float32
# )  # pre-allocate required memory for array for efficiency (4 bytes * sequences * products = big)
#
# BATCH_INDICES = np.arange(start=0, stop=len(X_test), step=BATCH_SIZE)  # row indices of batches
# BATCH_INDICES = np.append(BATCH_INDICES, len(X_test))  # add final batch_end row
#
# for index in np.arange(len(BATCH_INDICES) - 1):
#     batch_start = BATCH_INDICES[index]  # first row of the batch
#     batch_end = BATCH_INDICES[index + 1]  # last row of the batch
#     y_pred_probs[batch_start:batch_end] = model.predict_on_batch(X_test[batch_start:batch_end])

pred_time = time.time() - t_pred
print(
    "⏱️ Elapsed time for predicting {} odds for {} sequences: {:.4} seconds".format(
        N_TOP_ITEMS, len(y_test), pred_time
    )
)


####################################################################################################
# 🚀 EVALUATE RANKINGS
####################################################################################################


def extract_overlap_per_sequence(X_test, y_pred):
    """Finds overlapping items that are present in both arrays per row.
    Args:
        X_test (array): Input sequences for testing.
        y_pred (array): Predicted output sequences.
    Returns:
        list: A list of overlapping products for each row
    """
    overlap_items = [set(X_test[row, -5:]) & set(y_pred[row, :5]) for row in range(len(X_test))]
    return overlap_items


def compute_average_novelty(X_test, y_pred):
    """Computes the average overlap over all input and predicted sequences. Note that novelty is
    computed as 1 - overlap as the new items are the ones that are not present in both arrays.
    Args:
        X_test (array): Input sequences for testing.
        y_pred (array): Predicted output sequences.
    Returns:
        type: Average novelty over all predictions.
    """
    overlap_items = extract_overlap_per_sequence(X_test, y_pred)
    overlap_sum = np.sum([len(overlap_items[row]) for row in range(len(overlap_items))])
    average_novelty = 1 - (
        overlap_sum / (len(X_test) * X_test.shape[1])
    )  # new items are the ones that do not overlap
    return average_novelty


print("\n     Performance metrics on test set:")
y_pred = np.vstack(predicted_sequences[:, 0])  # top 1 recommendation (predicted next click)
gc.collect()

# TODO this ml_metric and vstack stuff can be implemented faster
accuracy = np.round(accuracy_score(y_test, y_pred), 4)
y_test = np.vstack(y_test)
map2 = np.round(average_precision.mapk(y_test, predicted_sequences, k=2), 4)
map4 = np.round(average_precision.mapk(y_test, predicted_sequences, k=4), 4)
map6 = np.round(average_precision.mapk(y_test, predicted_sequences, k=6), 4)
map12 = np.round(average_precision.mapk(y_test, predicted_sequences, k=12), 4)
coverage = np.round(len(np.unique(predicted_sequences[:, :4])) / len(np.unique(X_train)), 4)
novelty = np.round(compute_average_novelty(X_test[:, -4:], predicted_sequences[:, :4]), 4)

print("\n    Embedding GRU-RNN Product Type Ranker:")
print("     Accuracy @ 1   {:.4}%".format(accuracy * 100))
print("     MAP @ 2        {:.4}%".format(map2 * 100))
print("     MAP @ 4        {:.4}%".format(map4 * 100))
print("     MAP @ 6        {:.4}%".format(map6 * 100))
print("     MAP @ 12       {:.4}%".format(map12 * 100))
print("     Coverage       {:.4}%".format(coverage * 100))
print("     Novelty        {:.4}%".format(novelty * 100))

print("\n    Baseline Metrics:")
print("    Most Popular Items:")

pop_products = np.repeat([pop_products], axis=0, repeats=len(y_test))
accuracy_pop = np.round(accuracy_score(y_test, pop_products[:, -1:]), 4)
map2_pop = np.round(average_precision.mapk(y_test, pop_products, k=2), 4)
map4_pop = np.round(average_precision.mapk(y_test, pop_products, k=4), 4)
map6_pop = np.round(average_precision.mapk(y_test, pop_products, k=6), 4)
map12_pop = np.round(average_precision.mapk(y_test, pop_products, k=12), 4)
coverage_pop = np.round(len(np.unique(pop_products[:, :4])) / len(np.unique(X_train)), 4)
novelty_pop = np.round(compute_average_novelty(X_test[:, -4:], pop_products[:, :4]), 4)

print("     Accuracy @ 1   {:.4}%".format(accuracy_pop * 100))
print("     MAP @ 2        {:.4}%".format(map2_pop * 100))
print("     MAP @ 4        {:.4}%".format(map4_pop * 100))
print("     MAP @ 6        {:.4}%".format(map6_pop * 100))
print("     MAP @ 12       {:.4}%".format(map12_pop * 100))
print("     Coverage       {:.4}%".format(coverage_pop * 100))
print("     Novelty        {:.4}%".format(novelty_pop * 100))

print("\n    Last Viewed Items:")

accuracy_views = np.round(accuracy_score(y_test, X_test[:, -1:]), 4)
map2_views = np.round(average_precision.mapk(y_test, X_test[:, -2:], k=2), 4)
map4_views = np.round(average_precision.mapk(y_test, X_test[:, -4:], k=4), 4)
map6_views = np.round(average_precision.mapk(y_test, X_test[:, -6:], k=6), 4)
map12_views = np.round(average_precision.mapk(y_test, X_test[:, -12:], k=12), 4)
coverage_views = np.round(len(np.unique(X_test[:, -4:])) / len(np.unique(X_train)), 4)
novelty_views = np.round(compute_average_novelty(X_test, X_test[:, -4:]), 4)

print("     Accuracy @ 1   {:.4}%".format(accuracy_views * 100))
print("     MAP @ 2        {:.4}%".format(map2_views * 100))
print("     MAP @ 4        {:.4}%".format(map4_views * 100))
print("     MAP @ 6        {:.4}%".format(map6_views * 100))
print("     MAP @ 12       {:.4}%".format(map12_views * 100))
print("     Coverage       {:.4}%".format(coverage_views * 100))
print("     Novelty        {:.4}%".format(novelty_views * 100))

# plot model training history results
hist_dict = model_history.history
train_loss_values = hist_dict["loss"]
train_acc_values = hist_dict["accuracy"]
val_loss_values = hist_dict["val_loss"]
val_acc_values = hist_dict["val_accuracy"]
epochs = np.arange(1, len(model_history.history["loss"]) + 1).astype(int)

plt.close()
validation_plots, ax = plt.subplots(2, 1, figsize=(10, 6))
plt.subplot(211)  # plot loss over epochs
plt.plot(
    epochs, train_loss_values, "deepskyblue", linestyle="dashed", marker="o", label="Train Loss"
)
plt.plot(epochs, val_loss_values, "springgreen", marker="o", label="Val Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("{} loss over Epochs".format(model.loss).upper(), size=13, weight="bold")
plt.legend()

plt.subplot(212)  # plot accuracy over epochs
plt.plot(
    epochs, train_acc_values, "deepskyblue", linestyle="dashed", marker="o", label="Train Accuracy"
)
plt.plot(epochs, val_acc_values, "springgreen", marker="o", label="Val Accuracy")
plt.plot(epochs[-1], accuracy, "#16a085", marker="8", markersize=12, label="Test Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy (k=1) over Epochs".format(model.loss).upper(), size=13, weight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("./plots/validation_plots.png")

####################################################################################################
# 🚀 LOG EXPERIMENT
####################################################################################################

if LOGGING and not DRY_RUN:
    print("\n🧪 Logging experiment to mlflow")

    # Set tags
    mlflow.set_tags({"tf": tf.__version__, "machine": MACHINE})

    # Log parameters
    mlflow.log_param("n_items", N_TOP_ITEMS)
    mlflow.log_param("embed_dim", EMBED_DIM)
    mlflow.log_param("n_hidden_units", N_HIDDEN_UNITS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout", DROPOUT)
    mlflow.log_param("recurrent_dropout", RECURRENT_DROPOUT)
    mlflow.log_param("trainable_params", total_params)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("window", WINDOW_LEN)
    mlflow.log_param("pred_lookback", PRED_LOOKBACK)
    mlflow.log_param("min_items_train", MIN_ITEMS_TRAIN)
    mlflow.log_param("min_items_test", MIN_ITEMS_TEST)
    mlflow.log_param("shuffle_training", SHUFFLE_TRAIN_SET)
    mlflow.log_param("shuffle_val", SHUFFLE_TRAIN_AND_VAL_SET)
    mlflow.log_param("epochs", epochs[-1])
    mlflow.log_param("test_ratio", TEST_RATIO)
    mlflow.log_param("weeks_of_data", WEEKS_OF_DATA)
    mlflow.log_param("data_imbalance_correction", DATA_IMBALANCE_CORRECTION)
    mlflow.log_param("filter_repeated_uniques", FILTER_REPEATED_UNIQUES)
    mlflow.log_param("min_date", MIN_DATE)
    mlflow.log_param("max_date", MAX_DATE)

    # Log metrics
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("MAP 2", map2)
    mlflow.log_metric("MAP 4", map4)
    mlflow.log_metric("MAP 6", map6)
    mlflow.log_metric("MAP 12", map12)
    mlflow.log_metric("coverage", coverage)
    mlflow.log_metric("novelty", novelty)
    mlflow.log_metric("Train mins", np.round(train_time / 60), 2)
    mlflow.log_metric("Pred secs", np.round(pred_time))

    # Log artifacts
    mlflow.log_artifact("./gru_pt_keras_cloud.py")  # log executed code
    mlflow.log_artifact("./plots/validation_plots.png")  # log validation plots
    file = ".model_config.txt"  # log detailed model settings
    with open(file, "w") as model_config:
        model_config.write("{}".format(model.get_config()))
    mlflow.log_artifact("./model_config.txt")

    mlflow.end_run()

print("✅ All done, total elapsed time: {:.3} minutes".format((time.time() - t_prep) / 60))
gc.collect()
