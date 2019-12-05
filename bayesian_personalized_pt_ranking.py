
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp
import time

from tqdm import tqdm
from tensorflow.python.client import device_lib

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("notebook")

all_devices = str(device_lib.list_local_devices())
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if "GPU" in all_devices:
    DEVICE = "GPU"
    MACHINE = "Cloud VM"
elif "CPU" in all_devices:
    DEVICE = "CPU"
    MACHINE = "Local Machine"

print("🧠 Running TensorFlow version {} on {}".format(tf.__version__, DEVICE))

# model constants
epochs = 48
batches = 64
n_latent_vars = 64  # number of latent features for the matrix factorization
n_triplets = 4096  # how many (u,i,j) triplets we sample for each batch

# lambda regularization strength
lambda_id = 0.0
lambda_item = 0.0
lambda_bias = 0.0
learning_rate = 0.005

t_prep = time.time()  # start timer for preparing data

sequence_df = pd.read_csv('./data/ga_product_sequence_20191103.csv')
product_map_df = pd.read_csv('./data/product_mapping.csv')
sequence_df.head(3)

df = pd.DataFrame(sequence_df["product_sequence"].str.split(",").tolist(), index=sequence_df["coolblue_cookie_id"]).stack()
df = df.reset_index([0, "coolblue_cookie_id"])
df.columns = ['coolblue_cookie_id', 'product_id']
df['product_id'] = df['product_id'].astype(np.int)
df['product_type_id'] = df['product_id'].map(dict(zip(product_map_df["product_id"], product_map_df["product_type_id"])))
df.dropna(inplace=True)
df['product_type_id'] = df['product_type_id'].astype(np.int)
df.head(10)
df.drop('product_id', axis=1, inplace=True)
df = df.groupby(df.columns.tolist(), as_index=False).size()
df = df.reset_index(drop=False)
df.columns = ["id", "product_type_id", "clicks"]
df.head(10)

# Convert product_type_ids names into integer ids
df["cookie_token"] = df["id"].astype("category").cat.codes
df["product_token"] = df["product_type_id"].astype("category").cat.codes

# Create a lookup frame so we can get the product_type_ids back later
item_lookup = df[["product_token", "product_type_id"]].drop_duplicates()
item_lookup["product_token"] = item_lookup.product_token.astype(str)
df = df.drop(["id", "product_type_id"], axis=1)
df = df.loc[df.clicks != 0]  # drop sessions without views (contain no information)
df.head(3)

# lists of all ids, product_type_ids and clicks
ids = list(np.sort(df.cookie_token.unique()))
product_type_ids = list(np.sort(df.product_token.unique()))
clicks = list(df.clicks)

# rows and columns for our new matrix
rows = df.cookie_token.astype(float)
cols = df.product_token.astype(float)

# contruct a sparse matrix for our ids and items containing number of clicks
data_sparse = sp.csr_matrix((clicks, (rows, cols)), shape=(len(ids), len(product_type_ids)))
uids, iids = data_sparse.nonzero()

print("⏱️ Elapsed time for processing input data: {:.3} seconds".format(time.time() - t_prep))


graph = tf.Graph()


def init_variable(size, dim, name=None):
    """
    Helper function to initialize a new variable with uniform random values.
    """
    std = np.sqrt(2 / dim)
    return tf.Variable(tf.random_uniform([size, dim], -std, std), name=name)


def embed(inputs, size, dim, name=None):
    """
    Helper function to get a Tensorflow variable and create an embedding lookup to
    map our id and item indices to vectors.
    """
    emb = init_variable(size, dim, name)
    return tf.nn.embedding_lookup(emb, inputs)


def get_variable(graph, session, name):
    """
    Helper function to get the value of a Tensorflow variable by name.
    """
    v = graph.get_operation_by_name(name)
    v = v.values()[0]
    v = v.eval(session=session)
    return v


with graph.as_default():
    """
    Loss function: -SUM ln σ(xui - xuj) + λ(w1)**2 + λ(w2)**2 + λ(w3)**2
    ln = the natural log, σ(xuij) = the sigmoid function of xuij.
    λ = lambda regularization value, ||W||**2 = the squared L2 norm of our model parameters.
    """

    # Input into our model,  id (u), known item (i) an unknown item (i) triplets
    u = tf.placeholder(tf.int32, shape=(None, 1))
    i = tf.placeholder(tf.int32, shape=(None, 1))
    j = tf.placeholder(tf.int32, shape=(None, 1))

    # id feature embedding
    u_factors = embed(u, len(ids), n_latent_vars, "id_factors")  # U matrix

    # Known and unknown item embeddings
    item_factors = init_variable(len(product_type_ids), n_latent_vars, "item_factors")  # V matrix
    i_factors = tf.nn.embedding_lookup(item_factors, i)
    j_factors = tf.nn.embedding_lookup(item_factors, j)

    # i and j bias embeddings
    item_bias = init_variable(len(product_type_ids), 1, "item_bias")
    i_bias = tf.nn.embedding_lookup(item_bias, i)
    i_bias = tf.reshape(i_bias, [-1, 1])
    j_bias = tf.nn.embedding_lookup(item_bias, j)
    j_bias = tf.reshape(j_bias, [-1, 1])

    # Calculate the dot product + bias for known and unknown item to get xui and xuj
    xui = i_bias + tf.reduce_sum(u_factors * i_factors, axis=2)
    xuj = j_bias + tf.reduce_sum(u_factors * j_factors, axis=2)
    xuij = xui - xuj

    # Calculate the mean AUC (area under curve). If xuij is greater than 0, that means that xui is
    # greater than xuj (and thats what we want).
    u_auc = tf.reduce_mean(tf.cast(xuij > 0, float))
    tf.summary.scalar("auc", u_auc)

    # Calculate the squared L2 norm ||W||**2 multiplied by λ
    l2_norm = tf.add_n(
        [
            lambda_id * tf.reduce_sum(tf.multiply(u_factors, u_factors)),
            lambda_item * tf.reduce_sum(tf.multiply(i_factors, i_factors)),
            lambda_item * tf.reduce_sum(tf.multiply(j_factors, j_factors)),
            lambda_bias * tf.reduce_sum(tf.multiply(i_bias, i_bias)),
            lambda_bias * tf.reduce_sum(tf.multiply(j_bias, j_bias)),
        ]
    )

    # Calculate the loss as ||W||**2 - ln σ(Xuij)
    # loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
    loss = -tf.reduce_mean(tf.log(tf.sigmoid(xuij))) + l2_norm

    # Train using the Adam optimizer to minimize our loss function
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    step = opt.minimize(loss)
    init = tf.global_variables_initializer()


t_train = time.time()  # start timer for training
progress = tqdm(total=batches * epochs)  # progress bar for the training

session = tf.Session(config=None, graph=graph)
session.run(init)

for _ in range(epochs):
    for _ in range(batches):
        # randomly sample n_triplets indices
        idx = np.random.randint(low=0, high=len(uids), size=n_triplets)
        batch_u = uids[idx].reshape(-1, 1)
        batch_i = iids[idx].reshape(-1, 1)
        batch_j = np.random.randint(
            low=0, high=len(product_type_ids), size=(n_triplets, 1), dtype="int32"
        )  # randomly sample one unknown item for each id

        feed_dict = {u: batch_u, i: batch_i, j: batch_j}
        _, l, auc = session.run([step, loss, u_auc], feed_dict)

    progress.update(batches)
    progress.set_description("Loss: %.3f | AUC: %.3f" % (l, auc))

progress.close()

train_time = time.time() - t_train
print(
    "⏱️ Elapsed time for training on {} sequences: {:.3} minutes".format(len(df), train_time / 60)
)

# set datatypes
df['cookie_token'] = df['cookie_token'].astype(np.int32)
df['product_token'] = df['product_token'].astype(np.int32)
item_lookup['product_type_id'] = item_lookup['product_type_id'].astype(np.int32)
item_lookup['product_token'] = item_lookup['product_token'].astype(np.int32)

product_map_df = product_map_df.dropna()
product_map_df['product_type_id'] = product_map_df['product_type_id'].astype(np.int32)
product_map_df['subproduct_type_id'] = product_map_df['subproduct_type_id'].astype(np.int32)


def find_similar_product_type_ids(product_type_id=None, num_items=10):
    """Find product_type_ids similar to an product_type_id.
    Args:
        product_type_id (str): The name of the product_type_id we want to find similar product_type_ids for
        num_items (int): How many similar product_type_ids we want to return.
    Returns:
        similar (pandas.DataFrame): DataFrame with num_items product_type_id names and scores
    """

    id_vecs = get_variable(graph, session, "id_factors")  # matrix U
    item_vecs = get_variable(graph, session, "item_factors")  # matrix V
    item_bias = get_variable(graph, session, "item_bias").reshape(-1)
    item_id = int(item_lookup[item_lookup.product_type_id == product_type_id]["product_token"])
    item_vec = item_vecs[item_id].T  # Transpose item vector

    # calculate the similarity between this product and all other product_type_ids
    # by multiplying the item vector with our item_matrix
    scores = np.add(item_vecs.dot(item_vec), item_bias).reshape(1, -1)[0]
    top_10 = np.argsort(scores)[::-1][:num_items]  # Indices of top similarities

    # map the indices to product_type_id names
    product_type_ids, product_type_id_scores = [], []

    for idx in top_10:
        product_type_ids.append(
            item_lookup.product_type_id.loc[item_lookup.product_token == idx].iloc[0]
        )
        product_type_id_scores.append(scores[idx])

    similar = pd.DataFrame({"product_type_id": product_type_ids, "score": product_type_id_scores})
    similar["product_type_name"] = similar["product_type_id"].map(
        dict(zip(product_map_df["product_type_id"], product_map_df["product_type_name"]))
    )

    return similar


def make_recommendation(cookie_token=None, num_items=5):
    """Recommend items for a given id given a trained model
    Args:
        cookie_token (int): The id of the id we want to create recommendations for.
        num_items (int): How many recommendations we want to return.
    Returns:
        recommendations (pandas.DataFrame): DataFrame with num_items product_type_id names and scores
    """

    # make df of the session for input token
    clicks = df[df["cookie_token"] == cookie_token].merge(
        item_lookup, on="product_token", how="left"
    )
    clicks["product_type_name"] = clicks["product_type_id"].map(
        dict(zip(product_map_df["product_type_id"], product_map_df["product_type_name"]))
    )

    print("Making implicit feedback recommendations for observed user views: \n{}".format(clicks))
    print("\n----------------------⟶\n")

    id_vecs = get_variable(graph, session, "id_factors")  # matrix U
    item_vecs = get_variable(graph, session, "item_factors")  # matrix V
    item_bias = get_variable(graph, session, "item_bias").reshape(-1)
    rec_vector = np.add(id_vecs[cookie_token, :].dot(item_vecs.T), item_bias)
    item_idx = np.argsort(rec_vector)[::-1][:num_items]  # get indices of top cooki

    # map the indices to product_type_id names
    product_type_ids, scores = [], []

    for idx in item_idx:
        product_type_ids.append(
            item_lookup.product_type_id.loc[item_lookup.product_token == idx].iloc[0]
        )
        scores.append(rec_vector[idx])

    # add product information to recommendations
    recommendations = pd.DataFrame({"product_type_id": product_type_ids, "score": scores})
    recommendations["product_type_name"] = recommendations["product_type_id"].map(
        dict(zip(product_map_df["product_type_id"], product_map_df["product_type_name"]))
    )

    return recommendations


find_similar_product_type_ids(product_type_id=2675)  # wasmachines

find_similar_product_type_ids(product_type_id=17632)  # laptops

find_similar_product_type_ids(product_type_id=2093)  # mobiele telefoons

make_recommendation(cookie_token=512)

make_recommendation(cookie_token=1024)

make_recommendation(cookie_token=2048)

make_recommendation(cookie_token=4096)

make_recommendation(cookie_token=8192)


product_map_df.iloc[70000:700050]
