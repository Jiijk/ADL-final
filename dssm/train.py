import pandas as pd
import argparse
from models import HahowCouseModel
import tensorflow as tf
import tensorflow_recommenders as tfrs

# subgroups = pd.read_csv('data/subgroups.csv')
# groups = pd.read_csv('data/groups.csv')
test_seen = pd.read_csv('data/test_seen.csv')
test_unseen = pd.read_csv('data/test_unseen.csv')
test_seen_group = pd.read_csv('data/test_seen_group.csv')
test_unseen_group = pd.read_csv('data/test_unseen_group.csv')
train = pd.read_csv('data/train.csv')
train_group = pd.read_csv('data/train_group.csv')
val_seen = pd.read_csv('data/val_seen.csv')
val_unseen = pd.read_csv('data/val_unseen.csv')
val_seen_group = pd.read_csv('data/val_seen_group.csv')
val_unseen_group = pd.read_csv('data/val_unseen_group.csv')
courses = pd.read_csv('data/courses.csv')
users = pd.read_csv('data/users.csv')

user_features_vocabs_dict = pd.read_csv('data/user_features_vocabs_dict.csv')
course_features_vocabs_dict = pd.read_csv('data/course_features_vocabs_dict.csv')
train_dataset = pd.read_csv('data/train_dataset.csv').to_dict()
test_dataset = pd.read_csv('data/test_unseen_dataset.csv').to_dict()
val_dataset = pd.read_csv('data/val_unseen_dataset.csv').to_dict()

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--later_sizes', type=list, default=[64,128])
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--embedding_dim',type=int,default=64)
  parser.add_argument('--num_epochs', type=int, default=300)
  parser.add_argument('--lr', type=float, default=5e-4)
  parser.add_argument('val_batch_size',type=int,default=128)
  parser.add_argument('val_freq', type=int, default=20)

  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  configs = {
        'users':users, 
        'courses':courses ,
        'user_features_vocabs_dict':user_features_vocabs_dict, 
        'course_features_vocabs_dict':course_features_vocabs_dict, 
        'layer_sizes':args.layer_size,
        'embedding_dim':args.embedding_dim,
        'batch_size':args.batch_size
        }
  
  # contruct model with pre-process vocabs_dict
  model = HahowCouseModel(**configs)
  model.compile(optimizer=tf.keras.optimizers.Adagrad(args.lr))

  train_dataset_tf = tf.data.Dataset.from_tensor_slices(train_dataset)
  # test_dataset_tf = tf.data.Dataset.from_tensor_slices(test_dataset)
  val_dataset_tf = tf.data.Dataset.from_tensor_slices(val_dataset)

  cached_train = train_dataset_tf.batch(args.batch_size)
  cached_val = val_dataset_tf.batch(args.val_batch_size).cache()

  one_layer_history = model.fit(
      train_dataset,
      validation_data=cached_val,
      validation_freq=20,
      epochs=args.num_epochs,
      verbose=0)



