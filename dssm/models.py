import tensorflow as tf
import typing
import tensorflow_recommenders as tfrs


class UserModel(tf.keras.Model):
  def __init__(self, features_vocabs_dict:dict, embedding_dim=48):
    super().__init__()
    # features_vocabs_dict_tf = user_features_vocabs_dict_tf
    self.embedding_dim = embedding_dim
    self.features_vocabs_dict = features_vocabs_dict
    self.features_names = list(features_vocabs_dict.keys())
    self.features_vocabs_dict_tf = {}
    self.embedding_layers = {}
    self.look_up_dict = {}
    for name in self.features_names:
      self.features_vocabs_dict_tf = tf.data.Dataset.from_tensor_slices(self.features_vocabs_dict[name])
    for name in self.features_names:
      self.look_up_dict[name] =  tf.keras.layers.StringLookup()
      self.look_up_dict[name].adapt(self.features_vocabs_dict_tf[name])
      self.embedding_layers[name] = tf.keras.Sequential([
        self.look_up_dict[name],
        tf.keras.layers.Embedding(self.look_up_dict[name].vocab_size(),self.embedding_dim)
      ])
  def call(self, x_users):
    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    return tf.concat(
      [self.embedding_layers[name](x_users[name]) for name in self.features_names],
      axis=1)

class QueryModel(tf.keras.Model):
  """Model for encoding user queries."""
  def __init__(self, user_features_vocabs_dict, layer_sizes):
    super().__init__()
    # We first use the user model for generating embeddings.
    self.embedding_model = UserModel(user_features_vocabs_dict)
    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, x_users):
    feature_embedding = self.embedding_model(x_users)
    return self.dense_layers(feature_embedding)

class CourseModel(tf.keras.Model):
  def __init__(self, features_vocabs_dict,embedding_dim=48):
    super().__init__()

    max_tokens = 10_000
    self.embedding_dim = embedding_dim
    self.features_vocabs_dict = features_vocabs_dict
    self.features_names = list(features_vocabs_dict)
    self.features_vocabs_dict_tf = {}
    self.look_up_dict = {}
    self.embedding_layers = {}
    for name in self.features_names:
      self.features_vocabs_dict_tf = tf.data.Dataset.from_tensor_slices(self.features_vocabs_dict[name])

    for name in self.features_names:
      self.look_up_dict[name] =  tf.keras.layers.StringLookup()
      self.look_up_dict[name].adapt(self.features_vocabs_dict_tf[name])
      self.embedding_layers[name] = tf.keras.Sequential([
        self.look_up_dict[name],
        tf.keras.layers.Embedding(self.look_up_dict[name].vocab_size(),self.embedding_dim)
      ])

  def call(self, y_courses):
    return tf.concat(
      [self.embedding_layers[name](y_courses[name]) for name in self.features_names],
      axis=1)

class CandidateModel(tf.keras.Model):
  def __init__(self, course_features_vocabs_dict, layer_sizes):
    super().__init__()

    self.embedding_model = CourseModel(course_features_vocabs_dict)

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, y_courses):
    feature_embedding = self.embedding_model(courses)
    return self.dense_layers(feature_embedding)

class HahowCouseModel(tfrs.models.Model):

  def __init__(self, 
         users, 
         courses ,
         user_features_vocabs_dict, 
         course_features_vocabs_dict, 
         layer_sizes, 
         batch_size=128):
    super().__init__()
    self.users = users
    self.courses = courses
    self.batch_size = batch_size
    self.user_features_names = list(user_features_vocabs_dict.keys())
    self.course_features_names = list(course_features_vocabs_dict.keys())
    self.query_model = QueryModel(user_features_vocabs_dict, layer_sizes)
    self.candidate_model = CandidateModel(course_features_vocabs_dict, layer_sizes)
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=courses.batch(batch_size).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    query_embeddings = self.query_model(
        {name:self.users[name] for name in self.user_features_name})
    course_embeddings = self.candidate_model(
        {name:self.courses[name] for name in self.course_features_name})

    return self.task(
        query_embeddings, course_embeddings, compute_metrics=not training)

