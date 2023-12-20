import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import Dict, Text
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import mysql.connector


cnx = mysql.connector.connect(
    user='workbench-upabahasa-computer',
    password='4051ronadi',
    host='34.101.166.153',
    database='db_growumkm'
)

cursor = cnx.cursor()

query_tbl_userview = ("SELECT DISTINCT ID_user, ID_umkm from tbl_userview")
# query_tbl_umkm = ("SELECT DISTINCT ID_umkm from tbl_umkm")

cursor.execute(query_tbl_userview)
col_userview = ["ID_user", "ID_umkm"]
user = pd.DataFrame(cursor.fetchall(), columns=col_userview)
umkm = user["ID_umkm"].unique()
umkm = pd.DataFrame({"ID_umkm":umkm})

# Convert relevant object columns to strings in the UMKM dataset
user["ID_user"] = user["ID_user"].astype(str)
user["ID_umkm"] = user["ID_umkm"].astype(str)
umkm["ID_umkm"] = umkm["ID_umkm"].astype(str)

# Convert pandas DataFrames to TensorFlow tensors
def map_function(row):
    return {'ID_user': row['ID_user'],
          'ID_umkm': row['ID_umkm']}

ratings = tf.data.Dataset.from_tensor_slices(dict(user))
ratings = ratings.map(map_function)

umkm = tf.data.Dataset.from_tensor_slices(umkm["ID_umkm"])

ratings = ratings.map(lambda x: {
    "ID_umkm": x["ID_umkm"],
    "ID_user": x["ID_user"],
})
umkm = umkm.map(lambda x: x)


# Shuffle the dataset
shuffled = ratings.shuffle(buffer_size=len(ratings), seed=42)

# Take first 4000 elements for train
train = shuffled.take(4000)

# Skip 4000 elements and take next 320 for test
test = shuffled.skip(4000).take(320)


ID_umkm = umkm.batch(100)
ID_user = ratings.batch(1000).map(lambda x: x["ID_user"])

unique_ID_umkm = np.unique(np.concatenate(list(ID_umkm)))
unique_ID_user = np.unique(np.concatenate(list(ID_user)))

embedding_dimension = 32

user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_ID_user, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_ID_user) + 1, embedding_dimension)
])

umkm_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_ID_umkm, mask_token=None),
  tf.keras.layers.Embedding(len(unique_ID_umkm) + 1, embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
  candidates=umkm.batch(128).map(umkm_model)
)
task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class UmkmlensModel(tfrs.Model):

  def __init__(self, user_model, umkm_model):
    super().__init__()
    self.umkm_model: tf.keras.Model = umkm_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["ID_user"])
    # And pick out the umkm features and pass them into the umkm model,
    # getting embeddings back.
    positive_umkm_embeddings = self.umkm_model(features["ID_umkm"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_umkm_embeddings)

class NoBaseClassUmkmlensModel(tf.keras.Model):

  def __init__(self, user_model, umkm_model):
    super().__init__()
    self.umkm_model: tf.keras.Model = umkm_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Set up a gradient tape to record gradients.
    with tf.GradientTape() as tape:

      # Loss computation.
      user_embeddings = self.user_model(features["ID_user"])
      positive_umkm_embeddings = self.umkm_model(features["ID_umkm"])
      loss = self.task(user_embeddings, positive_umkm_embeddings)

      # Handle regularization losses as well.
      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    user_embeddings = self.user_model(features["ID_user"])
    positive_umkm_embeddings = self.umkm_model(features["ID_umkm"])
    loss = self.task(user_embeddings, positive_umkm_embeddings)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    user_embeddings = self.user_model(features["ID_user"])
    positive_umkm_embeddings = self.umkm_model(features["ID_umkm"])
    loss = self.task(user_embeddings, positive_umkm_embeddings)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

model = UmkmlensModel(user_model, umkm_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(1000).batch(100).cache()
cached_test = test.batch(100).cache()

model.fit(cached_train, epochs=3)

# Create a model that takes in raw query features, and
index_model = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index_model.index_from_dataset(
  tf.data.Dataset.zip((umkm.batch(100), umkm.batch(100).map(model.umkm_model)))
)


# Get the absolute path of the current file
current_file = os.path.abspath(__file__)

# Get the root project directory, by removing last part of the path
project_root = os.path.dirname(current_file)

# Model output directory under project root
model_dir = os.path.join(project_root, 'model')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save model to this dir
tf.saved_model.save(obj=model,
                    export_dir=model_dir,
                    signatures=tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)