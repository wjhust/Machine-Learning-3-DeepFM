#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import gzip
import logging
import math
import os
import random
import shutil
import time
from os import path, listdir
import numpy as np
import tensorflow as tf

from utils import input_reader_v2,tools,layers

__mtime__ = '2018/4/8'

'''
基于DeepInterestModelMultiLabel
新增特性：
1. user action field浅层特征利用
2. 为user action field的embedding增加attention pooling

'''


class DeepInterestModelMultiLabelV3:
    def __init__(self,
                 n_epoch,
                 batch_size,
                 embedding_dim,
                 nn_layer_shape,
                 attention_size,
                 feature_field_file,
                 num_parallel=10,
                 activation='relu',
                 learning_rate=0.001,
                 optimizer='adam',
                 steps_to_logout=1000,
                 need_dropout=False,
                 train_data_split=0.99,
                 train_end_date="",
                 train_day_length=15
                 ):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.nn_layer_shape = nn_layer_shape
        self.attention_size = attention_size 
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

        self.train_end_date = int(train_end_date)
        train_end_datetime = datetime.datetime.fromtimestamp(time.mktime(time.strptime(train_end_date, '%Y%m%d')))
        self.train_start_date = int(
            (train_end_datetime - datetime.timedelta(days=train_day_length - 1)).strftime("%Y%m%d"))
        self.train_day_length = train_day_length
        self.method = "DeepInterestModelMultiLabelV3"
        
#==========================================================================logging部分不太懂===============================================================
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename='logs/' + self.method + '.' + self.feature_field_file_name + '.log',
                            filemode='w')
        if train_end_date == "":
            logging.error('train_end_date is [' + train_end_date + '] is empty!')
            raise RuntimeError('train_end_date is [' + train_end_date + '] is empty!')
        logging.info(
            'n_epoch={} batch_size={} embedding_dim={} nn_layer_shape={} num_parallel={} activation={} learning_rate={} optimizer={} steps_to_logout={} sparse_field_name_list={} dense_field_name_list={} label_size={} need_dropout={} train_data_split={} train_end_date={} train_day_length={}'.format(
                str(self.n_epoch), str(self.batch_size), str(self.embedding_dim),
                ','.join(str(i) for i in self.nn_layer_shape), str(self.num_parallel), self.activation,
                str(self.learning_rate), self.optimizer, str(self.steps_to_logout),
                ','.join(self.sparse_field_name_list), ','.join(self.dense_field_name_list), str(self.label_size),
                self.need_dropout, str(self.train_data_split), str(self.train_end_date), str(self.train_day_length)))
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

#========================================================================================================================================================

    def load_index_dic(self, feature_field_file):#数据预处理，分类稀疏，稠密
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


#=========================================================================================================================================================

    def get_user_embedding_list(self, batch_parsed_features, need_dropout=False):
        embedding_list = []
        user_long_embedding_size = 0
        for field_name in self.sparse_field_name_list:
                field_sparse_ids = batch_parsed_features[field_name]
                field_sparse_values = batch_parsed_features[field_name + "_values"]#稀疏的输入进行embedding
                
                embedding = self.attention_layer_v2(self.tag_embedding_weight, field_sparse_ids,
                                                              field_sparse_values)
                embedding_list.append(embedding)
                user_long_embedding_size += self.embedding_dim


        #在embedding之前的dense层，先把待embedding的存储到lists当中
        for i, field_name in enumerate(self.dense_field_name_list): 
                field_dense_feature_values = tf.decode_raw(batch_parsed_features[field_name], tf.float32)
                embedding_list.append(field_dense_feature_values)
                user_long_embedding_size += self.dense_field_size[i]

        #整理embedding层，作为nn层的输入
        user_long_embedding = tf.concat(embedding_list, 1)
        user_long_embedding = tf.reshape(user_long_embedding, shape=[-1, user_long_embedding_size])
        print("user_long_embedding_size=" + str(user_long_embedding_size))
        
        
        
        #把embeddinglists输入到nn层，得到输出
        with tf.variable_scope("user_nn_layer"):
            input_layer_output = layers.get_nn_layer_v2(user_long_embedding, user_long_embedding_size,
                                                        self.nn_layer_shape,
                                                        activation=self.activation,
                                                        need_dropout=need_dropout)
        return input_layer_output
    
    
#======================================================================================================================================================    
    
    def test_op(self, batch_parsed_features):   #稀疏id，稀疏values的embedding
        embedding_list = []
        user_long_embedding_size = 0
        for field_name in self.sparse_field_name_list:
            with tf.variable_scope(field_name + "_embedding_layer", reuse=tf.AUTO_REUSE):
                field_sparse_ids = batch_parsed_features[field_name]
                field_sparse_values = batch_parsed_features[field_name + "_values"]
                embedding = self.attention_layer_v2(self.tag_embedding_weight, field_sparse_ids,
                                                              field_sparse_values)

        return field_sparse_ids,field_sparse_values, embedding
#======================================================================================================================================================
        
    def inference_op(self, batch_parsed_features): #获取imei和embedding_list
        imei = batch_parsed_features['imei']
        embedding_list = self.get_user_embedding_list(batch_parsed_features)

        return imei, embedding_list


#===================================================================================#比较重要================================================================================
        
    #将embedding权重和attention权重融合
    def attention_layer(self, embedding_matrix, sp_ids, sp_weights):
        dense_ids = tf.sparse_tensor_to_dense(sp_ids)                                   #[batch_size * (max(本次batch中每条样本的稀疏特征数))]
        max_feature_size = tf.shape(dense_ids)[1]                                       # max(本次batch中每条样本的稀疏特征数)
        dense_weights = tf.expand_dims(tf.sparse_tensor_to_dense(sp_weights), axis=-1)  # [batch_size * (max(本次batch中每条样本的稀疏特征数)),1]
        #tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引
        dense_embedding = tf.nn.embedding_lookup(embedding_matrix, dense_ids)           # [batch_size * (max(本次batch中每条样本的稀疏特征数)), embedding_size]
        dense_embedding_weighted = tf.multiply(dense_embedding, dense_weights)          # [batch_size ,max(本次batch中每条样本的稀疏特征数), embedding_size]
        dense_ids_mask = tf.expand_dims(tf.cast(dense_ids > 0, tf.float32),
                                        axis=-1)                                        # [batch_size * (max(本次batch中每条样本的稀疏特征数)),1]， 大于0取1.0，小于0取0


        x_inputs = tf.reshape(dense_embedding_weighted, shape=[-1, self.embedding_dim])  #[batch_size * max(本次batch中每条样本的稀疏特征数), embedding_size]
        x_inputs = tf.concat([x_inputs], axis=1)  
        #t f.contrib.layers.fully_connection(F，num_output,activation_fn)这个函数就是全链接成层,F是输入，num_output是下一层单元的个数，activation_fn是激活函数
        x_inputs = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=self.attention_size,
                                                    activation_fn=tf.sigmoid,
                                                    scope='attention_layer_%d' % i,
                                                    reuse = tf.AUTO_REUSE)
        
        attention_weight = tf.contrib.layers.fully_connected(inputs=x_inputs, num_outputs=1,
                                                            activation_fn=tf.sigmoid,
                                                            scope='attention_output',
                                                            reuse = tf.AUTO_REUSE)  # [(batch_size * max(本次batch中每条样本的稀疏特征数)),1]

        attention_weight = tf.reshape(attention_weight, shape=[-1, max_feature_size, 1])  # [(batch_size , max(本次batch中每条样本的稀疏特征数)),1]
        dense_embedding_weighted_attentioned = tf.multiply(dense_embedding_weighted, attention_weight)  # [batch_size ,max(本次batch中每条样本的稀疏特征数), embedding_size] * [(batch_size , max(本次batch中每条样本的稀疏特征数)),1]
        embedding_weighted_attentioned = tf.reduce_sum(tf.multiply(dense_embedding_weighted_attentioned, dense_ids_mask), 1)  # [batch_size, embedding_size]
        return embedding_weighted_attentioned



    #帮助处理稀疏输入
    def attention_layer_v2(self, embedding_matrix, sp_ids, sp_weights):
        with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
            dense_ids = tf.sparse_tensor_to_dense(sp_ids)   #[batch_size * (max(本次batch中每条样本的稀疏特征数))]
            max_feature_size = tf.shape(dense_ids)[1] #max(本次batch中每条样本的稀疏特征数)
            dense_weights = tf.expand_dims(tf.sparse_tensor_to_dense(sp_weights), axis=-1)  # [batch_size * (max(本次batch中每条样本的稀疏特征数)),1]
            dense_embedding = tf.nn.embedding_lookup(embedding_matrix, dense_ids)  # [batch_size * (max(本次batch中每条样本的稀疏特征数)), embedding_size]
            dense_embedding_weighted = tf.multiply(dense_embedding, dense_weights)   #[batch_size ,max(本次batch中每条样本的稀疏特征数), embedding_size]
            dense_ids_mask = tf.expand_dims(tf.cast(dense_ids > 0, tf.float32),
                                        axis=-1)  # [batch_size * (max(本次batch中每条样本的稀疏特征数)),1]， 大于0取1.0，小于0取0

            x_inputs = tf.reshape(dense_embedding_weighted, shape=[-1, max_feature_size, self.embedding_dim])   #[batch_size ,max(本次batch中每条样本的稀疏特征数), embedding_size]
            # Trainable parameters
            #get_variable获取已存在的变量（要求不仅名字，而且初始化方法等各个参数都一样），如果不存在，就新建一个。 可以用各种初始化方法，不用明确指定值。
            w_omega = tf.get_variable('w_omega',[self.embedding_dim, self.attention_size], 
                    initializer=tf.random_normal_initializer(stddev=(1 / math.sqrt(float(self.attention_size)))))
            b_omega = tf.get_variable('b_omega',[self.attention_size], 
                    initializer=tf.zeros_initializer)
            u_omega = tf.get_variable('u_omega',[self.attention_size], 
                    initializer=tf.random_normal_initializer(stddev=0.1))

            with tf.name_scope('v'):
                v = tf.tanh(tf.tensordot(x_inputs, w_omega, axes=1) + b_omega)   #v=w*x+b

            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape       vu=v*u
            alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
            output = tf.reduce_sum(x_inputs * tf.expand_dims(alphas, -1) * dense_ids_mask, 1)

            return output 

#===========================================================================================================================================================================================



    def train_op(self, batch_parsed_features):
        batch_labels = tf.sparse_tensor_to_dense(
            tf.sparse_merge(batch_parsed_features["label"], batch_parsed_features["label_values"], self.label_size))
        user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features, need_dropout=self.need_dropout)
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
        correct_prediction = tf.equal(tf.round(predictions), batch_labels)
        accuracy = tf.cast(correct_prediction, tf.float32)
        mean_accuracy = tf.reduce_mean(accuracy)
        tf.summary.scalar('mean_accuracy', mean_accuracy)
        # mean_average_precision = tf.metrics.average_precision_at_k(tf.cast(batch_labels,tf.int64),predictions,100)
        # tf.summary.scalar('mean_average_precision', mean_average_precision[0])
        return train_step, global_step, loss

    def map_op(self, batch_parsed_features):
        batch_labels = tf.sparse_tensor_to_dense(
            tf.sparse_merge(batch_parsed_features["label"], batch_parsed_features["label_values"], self.label_size))
        user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features, need_dropout=False)
        logits = tf.matmul(user_nn_layer_output, tf.transpose(self.tag_embedding_weight))
        logits = tf.nn.bias_add(logits, self.tag_embedding_biases)
        predictions = tf.nn.sigmoid(logits, name='prediction')
        return predictions, batch_labels

    def fit(self, tf_data_path):
        tf_data_files = []
        dir_cnt = 0

        logging.info("train_start_date:" + str(self.train_start_date))
        logging.info("train_end_date:" + str(self.train_end_date))

        for dir_name in listdir(tf_data_path):
            dir_name_i = int(dir_name)
            if dir_name_i < self.train_start_date or dir_name_i > self.train_end_date:
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
        validate_file_num = max(math.floor(len(tf_data_files) * (1 - self.train_data_split)), 1)

        # train_files = tf_data_files
        # validate_files = input_reader_v2.get_files('./parse_data_tools/data/tf_data_path/20180621')
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
                    '''
                    _ = sess.run(train_logit)
                    print(_)
                    break
                    '''
                    loss_sum += loss
                    steps_sum += 1
                    if global_step % self.steps_to_logout == 0:
                        # predictions,batch_labels = sess.run(map_logit)
                        result = sess.run(merged)
                        writer.add_summary(result, global_step)
                        # mAP = average_precision_score(batch_labels,predictions)
                        predictions, batch_labels = sess.run(validate_logit)
                        predictions = predictions[:, 1:]
                        batch_labels = batch_labels[:, 1:]
                        mAP = tools.calculate_mAP_v3(predictions, batch_labels, 100)
                        logging.info("mAP=" + str(mAP))
                        logging.info("train loss={}".format(loss_sum / steps_sum))
                        loss_sum = 0.0
                        steps_sum = 0
                        saver.save(sess, checkpoint_file, global_step=global_step)
            except tf.errors.OutOfRangeError:
                logging.info("End of dataset")
                saver.save(sess, checkpoint_file, global_step=global_step)
        writer.close()

    def user_inference(self, data_files, output_file):

        # data_files = input_reader_v2.get_files(data_path)
        next_element = input_reader_v2.get_input(data_files,
                                                 self.dense_field_name_list,
                                                 self.sparse_field_name_list,
                                                 self.num_parallel,
                                                 self.batch_size,
                                                 1,
                                                 buffer_size=self.batch_size * 10)
        #f_out = gzip.open(output_file, "wb")

        saver = tf.train.Saver()
        user_inference_op = self.inference_op(next_element)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, tf.train.latest_checkpoint("./checkpoint/DeepInterestModelMultiLabelV3"))
            all_vars = tf.trainable_variables()
            for p in all_vars:
                print(p.name,p.eval(sess))
            '''
            try:
                while True:
                    imei, logit = sess.run(user_inference_op)
                    imei = [str(i, encoding="utf-8") for i in imei]
                    logit = [sorted([(i, v) for i, v in enumerate(predicts)], key=lambda x: x[1], reverse=True)[0:100]
                             for predicts in logit]
                    logit = [' '.join([str(i[0]) + ':' + str(i[1]) for i in predicts]) for predicts in logit]
                    # logit = [' '.join([str(i[0]) for i in predicts]) for predicts in logit]
                    output = dict(zip(imei, logit))
                    for imei, predicts in output.items():
                        f_out.write(imei + ' ' + predicts + '\n')
                        f_out.flush()
            except tf.errors.OutOfRangeError:
                print("End of dataset")
            '''
    def get_item_embedding(self):
        return self.tag_embedding_weight, self.tag_embedding_biases

    def get_app_sim(self):
        tag_embedding_biases = tf.expand_dims(self.tag_embedding_biases,axis = -1)
        embedding_bias = tf.concat([self.tag_embedding_weight,tag_embedding_biases],axis=-1)
        embedding_bias_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(embedding_bias), axis=1)),axis=1)
        embedding_bias_norm = tf.matmul(embedding_bias_norm,tf.transpose(embedding_bias_norm))
        cosin_dis = tf.matmul(embedding_bias,tf.transpose(embedding_bias))/embedding_bias_norm
        app_sim_top_10 = tf.nn.top_k(cosin_dis, 1000)
        return app_sim_top_10.indices

    def dump_item_embedding(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("./checkpoint/DeepInterestModelMultiLabelV3"))
            tag_embedding_weight, tag_embedding_biases = sess.run(self.get_item_embedding())
            for i,embedding in enumerate(tag_embedding_weight):
                print(str(i) + ":" + ' '.join([str(score) for score in embedding])) 
                print(str(i) + ":" +str(tag_embedding_biases[i]))

    def dump_app_sim(self):
        saver = tf.train.Saver()
        np.set_printoptions(threshold=np.nan)
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("./checkpoint/DeepInterestModelMultiLabelV3"))
            sim_matrix = sess.run(self.get_app_sim())
            for app_list in sim_matrix:
                print(' '.join([str(i) for i in app_list]))
            #print(sim_matrix)
            '''
            for i,embedding in enumerate(sim_matrix):
                print(str(i) + ":" + ' '.join([str(score) for score in embedding])) 
                print(str(i) + ":" +str(tag_embedding_biases[i]))
            '''

    def get_attention_weight(self):
        with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
            #dense_ids = tf.expand_dims(tf.ones([self.label_size],dtype=tf.int32), axis=-1)
            #dense_weights = tf.expand_dims(tf.ones([self.label_size]), axis=-1)  
            sample_size = 101 #self.label_size 
            dense_ids = tf.constant(value=np.arange(sample_size),dtype=tf.int32)
            dense_ids = tf.expand_dims(dense_ids, axis=0)
            dense_weights = tf.expand_dims(tf.ones([sample_size]), axis=0)  
            dense_weights = tf.expand_dims(dense_weights, axis=-1)  # [batch_size * (max(本次batch中每条样本的稀疏特征数)),1]
            dense_embedding = tf.nn.embedding_lookup(self.tag_embedding_weight, dense_ids)
            dense_embedding_weighted = tf.multiply(dense_embedding, dense_weights)   #[batch_size ,max(本次batch中每条样本的稀疏特征数), embedding_size]
            dense_ids_mask = tf.expand_dims(tf.cast(dense_ids > 0, tf.float32),
                                        axis=-1)  # [batch_size * (max(本次batch中每条样本的稀疏特征数)),1]， 大于0取1.0，小于0取0
            max_feature_size = tf.shape(dense_ids)[1] #max(本次batch中每条样本的稀疏特征数)

            x_inputs = tf.reshape(dense_embedding_weighted, shape=[-1, max_feature_size, self.embedding_dim])   #[batch_size ,max(本次batch中每条样本的稀疏特征数), embedding_size]
            # Trainable parameters
            w_omega = tf.get_variable('w_omega',[self.embedding_dim, self.attention_size], 
                    initializer=tf.random_normal_initializer(stddev=(1 / math.sqrt(float(self.attention_size)))))
            b_omega = tf.get_variable('b_omega',[self.attention_size], 
                    initializer=tf.zeros_initializer)
            u_omega = tf.get_variable('u_omega',[self.attention_size], 
                    initializer=tf.random_normal_initializer(stddev=0.1))

            with tf.name_scope('v'):
                v = tf.tanh(tf.tensordot(x_inputs, w_omega, axes=1) + b_omega)

            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
            return alphas 

    def dump_attention_weight(self):
        get_attention_weight_op = self.get_attention_weight()
        saver = tf.train.Saver()
        np.set_printoptions(threshold=np.nan)
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint("./checkpoint/DeepInterestModelMultiLabelV3"))
            attention_weight = sess.run(get_attention_weight_op)
            #self.print_num_of_total_parameters(output_detail=True)
            print(attention_weight)
    def print_num_of_total_parameters(self, output_detail=False, output_to_logging=False):
        total_parameters = 0
        parameters_string = ""

        for variable in tf.all_variables():

            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
            if variable.name != "attention_layer_input/weights:0":
                continue
            if len(shape) == 1:
                parameters_string += ("%s %d, " % (variable.name, variable_parameters))
            else:
                parameters_string += ("%s %s=%d, " % (variable.name, str(shape), variable_parameters))

            print(variable)
        if output_to_logging:
            if output_detail:
                logging.info(parameters_string)
            logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
        else:
            if output_detail:
                print(parameters_string)
            print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
