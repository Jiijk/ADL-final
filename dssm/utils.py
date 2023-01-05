import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import genfromtxt
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.types.doc_typealias import Dict


# x_train = genfromtxt('data/x_train.csv', delimiter=',')
# y_train = genfromtxt('data/y_train.csv', delimiter=',')
# x_test = genfromtxt('data/x_test.csv', delimiter=',')
# y_test = genfromtxt('data/y_test.csv', delimiter=',').T

# # subgroups = pd.read_csv('data/subgroups.csv')
# sub_groups = pd.read_csv('data/sub_interests.csv')
# groups = pd.read_csv('data/groups.csv')
# test_seen = pd.read_csv('data/test_seen.csv')
# test_unseen = pd.read_csv('data/test_unseen.csv')
# test_seen_group = pd.read_csv('data/test_seen_group.csv')
# test_unseen_group = pd.read_csv('data/test_unseen_group.csv')
# train = pd.read_csv('data/train.csv')
# train_group = pd.read_csv('data/train_group.csv')
# val_seen = pd.read_csv('data/val_seen.csv')
# val_unseen = pd.read_csv('data/val_unseen.csv')
# val_seen_group = pd.read_csv('data/val_seen_group.csv')
# val_unseen_group = pd.read_csv('data/val_unseen_group.csv')
# courses = pd.read_csv('data/courses.csv')
# users = pd.read_csv('data/users.csv')
# x_train_num = pd.read_csv('data/x_train_num.csv')
# y_train_num = pd.read_csv('data/y_train_num.csv')
# x_test_num = pd.read_csv('data/x_test_num.csv')
# y_test_num = pd.read_csv('data/y_test_num.csv')

##### deprecated data preprocess
"""
Some data preprocess been deprecated, but may be a necessary sub routine for below function.
Feature choosen:interests(groups+subgroups)
catch exception element, for example:no interests user
dict interests_groups[groups]:subgroups
subgroups list interests
dict type user_interests[user]:subgroups
 
"""
# interests_groups = {}
# interests = []
# no_int_users = []
# users_interests = {}
# for index,interest in users['interests'].iteritems():
#   user_id = users.loc[index,'user_id']
#   if isinstance(interest,float):
#     no_int_users.append(index)
#     continue
#   interest = interest.split(',')
#   for group in interest:
#     self_interests = []
#     group = group.split('_')
#     for _int in group:
#       self_interests.append(_int)
#       if _int not in interests:
#         interests.append(_int)
#     if group[0] not in interests_groups.keys():
#       interests_groups[group[0]] = [group[1]]
#     elif group[1] not in interests_groups[group[0]]:
#       interests_groups[group[0]].append(group[1])
#   users_interests[user_id] = self_interests

# # x.shape = [#users, #interests]
# x_ints = np.zeros((len(users), len(interests)))
# for idx,_interests in users['interests'].iteritems():
#   if idx not in no_int_users:
#     _interests = _interests.split(',')
#     _interests = [group.split('_') for group in _interests]
#     _interests = [_int for group in _interests for _int in group]
#     for _int in _interests:
#       x_ints[idx][interests.index(_int)] = 1

# occupation_titles = []
# no_occ_users = []
# for index,occs in users['occupation_titles'].iteritems():
#   if isinstance(occs,float):
#     no_occ_users.append(index)
#     continue
#   occs = occs.split(',')
#   for occ in occs:
#     if occ not in occupation_titles:
#       occupation_titles.append(occ)
# print(len(occupation_titles))

# recreation_names = []
# no_rec_users = []
# for index,recs in users['recreation_names'].iteritems():
#   if isinstance(recs,float):
#     no_rec_users.append(index)
#     continue
#   recs = recs.split(',')
#   for rec in recs:
#     if rec not in recreation_names:
#       recreation_names.append(rec)
# # print(len(recreation_names)) 


# def user2feature_vocabs(user:pd.DataFrame()) ->dict:
#   user_features_names = np.array(['user_id','gender','interests','occupation_titles','recreation_names'])
#   user_features_vocabs_dict = {}
#   user_features_collection = []
#   user_ids = np.array(users['user_id'].tolist())
#   unique_ids = np.unique(user_ids)
#   user_features_vocabs_dict['user_id'] = unique_ids
#   user_features_vocabs_dict['gender'] = np.array([0,1,2])

#   for feature_name in user_features_names:
#     user_features_collection.append(users[feature_name])

#   temp = [interests,occupation_titles,recreation_names]
#   for i,feature in enumerate(user_features_names[2:]):
#     user_features_vocabs_dict[feature] = np.array(temp[i])
#   return user_features_vocabs_dict

# def course2feature_vocab_dict(course:pd.DataFrame) ->dict:
#   course_features_names = np.array(['course_id','course_name','course_price','teacher_id','groups','sub_groups'])
#   course_features_collection = []
#   course_features_vocabs_dict = {}
#   for feature in course_features_names[:4]:
#     course_feat = np.array(courses[feature].tolist())
#     unique_feat = np.unique(course_feat)
#     course_features_vocabs_dict[feature] = unique_feat

#   temp = [groups, sub_groups]
#   for i,feature in enumerate(course_features_names[4:]):
#     course_features_vocabs_dict[feature] = np.array(temp[i])
#   return course_features_vocabs_dict

# def extend_train(train:pd.DataFrame, course_features=None) -> pd.DataFrame:
#   # dict_keys(['user_id', 'gender', 'interests', 'occupation_titles', 'recreation_names'])
#   course_features = ['course_id','course_name','course_price','teacher_id','groups','sub_groups']
#   x_train = pd.DataFrame(columns=course_features)
#   base_index = len(train)
#   count = 0
#   for index, user in train.iterrows():
#     courses_ids = np.array(train['course_id'][index].split(' '))
#     if len(courses_ids) > 1:
#       original_block = pd.DataFrame({'user_id':user['user_id'],'course_id':courses_ids[0]},index=[0])
#       train.loc[index] = original_block.loc[0,['user_id','course_id']]
#       for course in courses_ids[1:]:
#         extend_block = pd.DataFrame({'user_id':user['user_id'],'course_id':course},index=[base_index+count])
#         train = pd.concat((train,extend_block))
#         count += 1
#   for index,user in train.iterrows():
#     self_interests = []
#     course = courses[courses['course_id'].str.match(user['course_id'])]
#     groups = course['groups'].item()
#     sub_groups = course['sub_groups'].item()
#     if not isinstance(groups, float):
#       self_interests.append(groups)
#     if not isinstance(sub_groups, float):
#       for _int in course['sub_groups'].item().split(','):
#         self_interests.append(_int)    
#     extend_block = pd.concat(course[course_features[2:]],axis=1)
#     orginal_block = pd.DataFrame({'course_id':user[0],'user_id':user[1]},index=[0])
#     extend_block = pd.concat((orginal_block, extend_block),axis=1)
#     x_train.loc[index] = extend_block.loc[0,extend_block.columns]
#   return x_train