#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
import os
import random
import shutil
from os import path, listdir
import time
import datetime
import tensorflow as tf
import numpy as np

from utils import input_reader_v2
from utils import layers
from utils import tools

__mtime__ = '2018/4/8'

class DeepInterestModelMultiLabel:
    def __init__(self,
                 n_epoch,
                 batch_size,
                 embedding_dim,
                 nn_layer_shape,
                 feature_field_file,
                 num_parallel=10,
                 activation='relu',
                 learning_rate=0.001,
                 optimizer='adam',
                 steps_to_logout=1000,
                 need_dropout=False,
                 train_data_split=0.99,
                 train_end_date="",
                 train_day_length=15,
                 max_steps=10000
                 ):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.nn_layer_shape = nn_layer_shape
        self.num_parallel = num_parallel
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.steps_to_logout = steps_to_logout
        self.sparse_field_name_list, \
        self.dense_field_name_list, \
        self.dense_field_size, \
        self.label_size = self.load_index_dic(feature_field_file)
        self.feature_field_file_name = path.basename(feature_field_file)
        self.need_dropout = need_dropout
        self.train_data_split = train_data_split
        self.max_steps = max_steps
        
        '''    
        indexs = tf.placeholder(tf.int64, [None, 2])
        ids = tf.placeholder(tf.int64, [None])
        values = tf.placeholder(tf.float32, [None])
        dense_str = tf.placeholder(tf.string, [None])
        feature_shape = tf.placeholder(tf.int64, [2])

        self.input_placeholder = {'imei':SparseTensor(indexs,ids,feature_shape),'label_values':SparseTensor(indexs,values,feature_shape),'imei':dense_str}
        for field in self.sparse_field_name_list:
            self.input_placeholder[field + "_values"] = SparseTensor(indexs,values,feature_shape) 
            self.input_placeholder[field + "_values_float"] = SparseTensor(indexs,values,feature_shape) 
            self.input_placeholder[field] = SparseTensor(indexs,ids,feature_shape) 

        for field in self.dense_field_name_list:
            self.input_placeholder[field] = dense_str 
        '''
        
#=====================================================================================================================================================================================        
        self.train_end_date = int(train_end_date)
        train_end_datetime = datetime.datetime.fromtimestamp(time.mktime(time.strptime(train_end_date, '%Y%m%d')))
        self.train_start_date = int((train_end_datetime-datetime.timedelta(days=train_day_length-1)).strftime("%Y%m%d"))
        self.train_day_length = train_day_length
        self.method = "DeepInterestModelMultiLabel"
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename='logs/' + self.method + '.' + self.feature_field_file_name + '.'  + str(self.embedding_dim) + '.log',
                            filemode='w')
        if train_end_date == "":
            logging.error('train_end_date is [' + train_end_date + '] is empty!')
            raise RuntimeError('train_end_date is [' + train_end_date + '] is empty!')
        logging.info('n_epoch={} batch_size={} embedding_dim={} nn_layer_shape={} num_parallel={} activation={} learning_rate={} optimizer={} steps_to_logout={} sparse_field_name_list={} dense_field_name_list={} label_size={} need_dropout={} train_data_split={} train_end_date={} train_day_length={}'.format(str(self.n_epoch), str(self.batch_size),str(self.embedding_dim),','.join(str(i) for i in self.nn_layer_shape),str(self.num_parallel),self.activation,str(self.learning_rate),self.optimizer,str(self.steps_to_logout),','.join(self.sparse_field_name_list),','.join(self.dense_field_name_list),str(self.label_size),self.need_dropout,str(self.train_data_split),str(self.train_end_date),str(self.train_day_length)))
        with tf.variable_scope("tag_embedding_layer", reuse=tf.AUTO_REUSE):
            self.tag_embedding_weight = tf.get_variable("tag_embedding",
                    [self.label_size, self.embedding_dim],
                    initializer=tf.random_normal_initializer(
                        stddev=(1 / math.sqrt(float(self.embedding_dim)))),
                    trainable=True)
            self.tag_embedding_biases = tf.get_variable("tag_embedding_biases",
                    [self.label_size],
                    initializer=tf.zeros_initializer,
                    trainable=True)

            self.tag_ones = tf.expand_dims(tf.ones([self.label_size]), dim=1)
#=====================================================================================================================================================================================
            
    def load_index_dic(self, feature_field_file):#稀疏，稠密，label分类-----预处理
        print(feature_field_file)
        f_field_data = open(feature_field_file)
        sparse_field_name_list = []
        dense_field_name_list = []
        dense_field_size = []
        label_size = 0
        for line in f_field_data:
            line = line.strip('\n').strip('\r')
            line_arr = line.split(' ')
            if line_arr[0] == 'sparse':
                sparse_field_name_list.append(line_arr[1])
            elif line_arr[0] == 'dense':
                dense_field_name_list.append(line_arr[1])
                dense_field_size.append(int(line_arr[2]))
            elif line_arr[0] == 'label_size':
                label_size = int(line_arr[1])
        f_field_data.close()
        return sparse_field_name_list, dense_field_name_list, dense_field_size, label_size

#==========================================================================================================================================================================================

    def get_user_embedding_list(self, batch_parsed_features, combiner='mean',need_dropout=False):#
        embedding_list = []
        user_long_embedding_size = 0
        for field_name in self.sparse_field_name_list:
            with tf.variable_scope(field_name + "_embedding_layer", reuse=tf.AUTO_REUSE):#embedding层，自动实现共享变量 
                field_sparse_ids = batch_parsed_features[field_name]
                field_sparse_values = batch_parsed_features[field_name + "_values"]
                if combiner == 'mean':
                    embedding = tf.nn.embedding_lookup_sparse(self.tag_embedding_weight, field_sparse_ids,
                                                              field_sparse_values,
                                                              combiner="mean")
                elif combiner == 'avg':
                    embedding = tf.nn.embedding_lookup_sparse(self.tag_embedding_weight, field_sparse_ids,
                                                              field_sparse_values,
                                                              combiner="sum")
                    
                    sparse_features = tf.sparse_merge(field_sparse_ids, field_sparse_values, vocab_size=self.label_size)
                    
                    sparse_x_feature_cnt = tf.sparse_tensor_dense_matmul(sparse_features, self.tag_ones)#（稠密矩阵）self.tag_ones*sparse_features（稀疏矩阵）
                    embedding = tf.div(embedding, sparse_x_feature_cnt)  #  embedding/（self.tag_ones*sparse_features）
                embedding_list.append(embedding)
                user_long_embedding_size += self.embedding_dim

        for i, field_name in enumerate(self.dense_field_name_list):
            with tf.variable_scope(field_name + "_dense_layer", reuse=tf.AUTO_REUSE):#dense层，进行压缩
                field_dense_feature_values = tf.decode_raw(batch_parsed_features[field_name], tf.float32)
                embedding_list.append(field_dense_feature_values)
                user_long_embedding_size += self.dense_field_size[i]

        user_long_embedding = tf.concat(embedding_list, 1)
        user_long_embedding = tf.reshape(user_long_embedding, shape=[-1, user_long_embedding_size])
        print("user_long_embedding_size=" + str(user_long_embedding_size))
        
        
        with tf.variable_scope("user_nn_layer"):#nn层
            input_layer_output = layers.get_nn_layer_v2(user_long_embedding, user_long_embedding_size,
                    self.nn_layer_shape,
                    activation=self.activation,
                    need_dropout=need_dropout)
        return input_layer_output


#==============================================================================================================================================================================

    def test_op(self, batch_parsed_features):
        embedding_list = []
        user_long_embedding_size = 0
        combiner = 'mean'
        for field_name in self.sparse_field_name_list:
            with tf.variable_scope(field_name + "_embedding_layer", reuse=tf.AUTO_REUSE): #embedding 层
                field_sparse_ids = batch_parsed_features[field_name]
                field_sparse_values = batch_parsed_features[field_name + "_values"]
                if combiner == 'mean':
                    embedding = tf.nn.embedding_lookup_sparse(self.tag_embedding_weight, field_sparse_ids,
                                                              field_sparse_values,
                                                              combiner="mean")
                elif combiner == 'avg':
                    embedding = tf.nn.embedding_lookup_sparse(self.tag_embedding_weight, field_sparse_ids,
                                                              field_sparse_values,
                                                              combiner="sum")
                    sparse_features = tf.sparse_merge(field_sparse_ids, field_sparse_values, vocab_size=self.label_size)
                    sparse_x_feature_cnt = tf.sparse_tensor_dense_matmul(sparse_features, self.tag_ones)
                    embedding = tf.div(embedding, sparse_x_feature_cnt)
                embedding_list.append(embedding)
                user_long_embedding_size += self.embedding_dim

        for i, field_name in enumerate(self.dense_field_name_list):
            with tf.variable_scope(field_name + "_dense_layer", reuse=tf.AUTO_REUSE):
                field_dense_feature_values = tf.decode_raw(batch_parsed_features[field_name], tf.float32)
                embedding_list.append(field_dense_feature_values)
                user_long_embedding_size += self.dense_field_size[i]

        user_long_embedding = tf.concat(embedding_list, 1)
        print(str(user_long_embedding_size))
        return user_long_embedding 

    def inference_op(self, batch_parsed_features):#返回的为imei，以及nn的输出
        imei = batch_parsed_features['imei']
        user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features,need_dropout=False)
        return imei, user_nn_layer_output 

    def train_op(self, batch_parsed_features):
        batch_labels = tf.sparse_tensor_to_dense(
            tf.sparse_merge(batch_parsed_features["label"], batch_parsed_features["label_values"], self.label_size))
        user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features,need_dropout=self.need_dropout)
        logits = tf.matmul(user_nn_layer_output, tf.transpose(self.tag_embedding_weight))
        logits = tf.nn.bias_add(logits, self.tag_embedding_biases)
        # train loss
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=logits)
        # cross_entropy = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=batch_labels)
        loss = tf.reduce_mean(cross_entropy)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)
        tf.summary.scalar('loss', loss)

        # accuracy
        predictions = tf.nn.sigmoid(logits, name='prediction')
        #tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
        #tf.round 将张量的值四舍五入为最接近的整数，元素
        correct_prediction = tf.equal(tf.round(predictions), batch_labels)
        #tf.cast(x, dtype, name=None)  将输入x转化为dtype型
        accuracy = tf.cast(correct_prediction, tf.float32)
        #tf.reduce_mean求均值
        mean_accuracy = tf.reduce_mean(accuracy)
        tf.summary.scalar('mean_accuracy', mean_accuracy)
        # mean_average_precision = tf.metrics.average_precision_at_k(tf.cast(batch_labels,tf.int64),predictions,100)
        # tf.summary.scalar('mean_average_precision', mean_average_precision[0])
        return train_step, global_step,loss 




    def map_op(self, batch_parsed_features):
        batch_labels = tf.sparse_tensor_to_dense(
            tf.sparse_merge(batch_parsed_features["label"], batch_parsed_features["label_values"], self.label_size))
        
        user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features,need_dropout=False)
        logits = tf.matmul(user_nn_layer_output, tf.transpose(self.tag_embedding_weight))
        logits = tf.nn.bias_add(logits, self.tag_embedding_biases)
        predictions = tf.nn.sigmoid(logits, name='prediction')#将输出sigmoid后输出
        return predictions, batch_labels





#======================================================================================看不懂===========================================================================================================
    def fit(self, tf_data_path):
        tf_data_files = []
        dir_cnt = 0


        logging.info("train_start_date:" + str(self.train_start_date))
        logging.info("train_end_date:" + str(self.train_end_date))


        for dir_name in listdir(tf_data_path):
            dir_name_i = int(dir_name)
            if dir_name_i < self.train_start_date or dir_name_i > self.train_end_date:#不在训练时间内
                continue
            dir_cnt += 1
            data_path = path.join(tf_data_path, dir_name)
            if path.isdir(data_path):
                _data_files = input_reader_v2.get_files(data_path)
                if len(_data_files) > 0:
                    tf_data_files.extend(_data_files)
        logging.info("train data cnt is " + str(dir_cnt))
        
        if dir_cnt != self.train_day_length:
            logging.error('train data is less than ' + str(self.train_day_length))
            raise RuntimeError('train data is less than ' + str(self.train_day_length))
            
        random.shuffle(tf_data_files)
        validate_file_num = max(math.floor(len(tf_data_files) * (1-self.train_data_split)), 1)

        #train_files = tf_data_files
        #validate_files = input_reader_v2.get_files('./parse_data_tools/data/tf_data_path/20180621')
        train_files = tf_data_files[:-validate_file_num]
        validate_files = tf_data_files[-validate_file_num:]
        logging.info("train_files : {}".format(','.join(train_files)))
        logging.info("validate_files : {}".format(','.join(validate_files)))
        next_element = input_reader_v2.get_input(train_files,
                                                 self.dense_field_name_list,
                                                 self.sparse_field_name_list,
                                                 self.num_parallel,
                                                 self.batch_size,
                                                 self.n_epoch,
                                                 buffer_size=self.batch_size * 10)

        train_logit = self.train_op(next_element)
        #train_logit = self.test_op(next_element)
        
        validate_element = input_reader_v2.get_input(validate_files,
                                                     self.dense_field_name_list,
                                                     self.sparse_field_name_list,
                                                     self.num_parallel,
                                                     self.batch_size,
                                                     self.n_epoch,
                                                     buffer_size=self.batch_size)
        validate_logit = self.map_op(validate_element)
        #validate_logit = self.test_op(validate_element)
        inference_op = self.inference_op(self.input_placeholder)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())

        checkpoint_path = "./checkpoint/" + self.method + "/"
        try:
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        except Exception as e:
            logging.info("Fail to remove checkpoint_path, exception: {}".format(e))
        os.makedirs(checkpoint_path)
        checkpoint_file = path.join(checkpoint_path, "checkpoint.ckpt")
        saver = tf.train.Saver(max_to_keep=10)
        np.set_printoptions(threshold=np.nan)
        with tf.Session() as sess:
            sess.run(init_op)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./summary/' + self.method, graph=tf.get_default_graph(),
                                           filename_suffix='_' + self.feature_field_file_name + '_' + '_'.join(
                                               str(i) for i in self.nn_layer_shape))
            logging.info("Start train")
            loss_sum = 0.0
            steps_sum = 0
            try:
                while True:
                    train_step, global_step, loss = sess.run(train_logit)
                    #_ = sess.run(train_logit)
                    #print(_.shape)
                    loss_sum += loss
                    steps_sum += 1
                    if global_step % self.steps_to_logout == 0:
                        # predictions,batch_labels = sess.run(map_logit)
                        result = sess.run(merged)
                        writer.add_summary(result, global_step)
                        # mAP = average_precision_score(batch_labels,predictions)
                        predictions, batch_labels = sess.run(validate_logit)
                        predictions = predictions[:,1:]
                        batch_labels = batch_labels[:,1:]
                        mAP = tools.calculate_mAP_v3(predictions, batch_labels, 100)
                        logging.info("mAP=" + str(mAP))
                        logging.info("train loss={}".format(loss_sum / steps_sum))
                        loss_sum = 0.0
                        steps_sum = 0
                        saver.save(sess, checkpoint_file, global_step=global_step)
                    if self.max_steps > 0 and global_step > self.max_steps:
                        break
            except tf.errors.OutOfRangeError:
                logging.info("End of dataset")
            saver.save(sess, checkpoint_file, global_step=global_step)
        writer.close()




    def user_inference(self, data_path):
        #tf.reset_default_graph()
        data_files = input_reader_v2.get_files(data_path)
        next_element = input_reader_v2.get_input(data_files,
                                                 self.dense_field_name_list,
                                                 self.sparse_field_name_list,
                                                 self.num_parallel,
                                                 self.batch_size,
                                                 1,
                                                 buffer_size=self.batch_size * 10)
        #saver = tf.train.import_meta_graph("./checkpoint/DeepInterestModelMultiLabel/checkpoint.ckpt-384576.meta")
        #train_op = self.train_op(next_element)
        
        user_inference_op = self.inference_op(next_element)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            #sess.run(init_op)
            saver.restore(sess, tf.train.latest_checkpoint("./checkpoint/DeepInterestModelMultiLabel"))
            try:
                while True:
                    imei, logit = sess.run(user_inference_op)
                    imei = [str(i, encoding="utf-8") for i in imei]
                    logit = [sorted([(i, v) for i, v in enumerate(predicts)], key=lambda x: x[1], reverse=True)[0:100]
                             for predicts in logit]
                    #logit = [' '.join([str(i[0]) + ':' + str(i[1]) for i in predicts]) for predicts in logit]
                    logit = [' '.join([str(i[0]) for i in predicts]) for predicts in logit]
                    output = dict(zip(imei, logit))
                    for imei, predicts in output.items():
                        print(imei + ' ' + predicts)
                    break
            except tf.errors.OutOfRangeError:
                print("End of dataset")


#=============================================================================================================================================================================



    def get_app_sim(self):
        #维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数，-1表示在最后加一维
        tag_embedding_biases = tf.expand_dims(self.tag_embedding_biases,axis = -1)
        embedding_bias = tf.concat([self.tag_embedding_weight,tag_embedding_biases],axis=-1)
        
        embedding_bias_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(embedding_bias), axis=1)),axis=1) #  tensor=sqrt(sigma( bias^2) )
        embedding_bias_norm = tf.matmul(embedding_bias_norm,tf.transpose(embedding_bias_norm))    #  tensor * tensor^T
        cosin_dis = tf.matmul(embedding_bias,tf.transpose(embedding_bias))/embedding_bias_norm    #  embedding_bias * embedding_bias^T  /  (sigma( bias^2) )
        app_sim_top_10 = tf.nn.top_k(cosin_dis, 10000) #tf.nn.top_k(input, k, name=None)返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
        return app_sim_top_10.indices

    def dump_app_sim(self):
        saver = tf.train.Saver()
        #set_printoptions来强制NumPy打印所有数据np.set_printoptions(threshold='nan')
        np.set_printoptions(threshold=np.nan)
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("./checkpoint/DeepInterestModelMultiLabel"))
            sim_matrix = sess.run(self.get_app_sim())
            print(sim_matrix)
