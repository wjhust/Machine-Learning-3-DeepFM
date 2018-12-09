#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
import os
import random
import shutil
from os import path

import tensorflow as tf

from utils import input_reader_v2
from utils import layers,tools

__mtime__ = '2018/4/8'


class DeepInterestModelMultiClass:
    def __init__(self,
                 n_epoch,
                 batch_size,
                 embedding_dim,
                 nn_layer_shape,
                 feature_field_file,
                 label_size,
                 num_parallel=10,
                 activation='relu',
                 learning_rate=0.001,
                 optimizer='adam',
                 steps_to_logout=1000,
                 need_dropout=False,
                 train_data_split=0.99,
                 softmax_sampled_cnt=1000,
                 l2_reg=0.001#L2正则化
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
        self.softmax_sampled_cnt = softmax_sampled_cnt
        self.label_size = label_size
        self.l2_reg = l2_reg


#========================================================logging部分，不太懂============================================================
        self.method = 'DeepInterestModelMultiClass'
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename='logs/' + self.method + '.' + self.feature_field_file_name + '.log',
                            filemode='w')
        logging.info(
            'n_epoch={} batch_size={} embedding_dim={} nn_layer_shape={} num_parallel={} activation={} learning_rate={} optimizer={} steps_to_logout={} sparse_field_name_list={} dense_field_name_list={} label_size={} need_dropout={} train_data_split={}'.format(
                str(self.n_epoch), str(self.batch_size), str(self.embedding_dim),
                ','.join(str(i) for i in self.nn_layer_shape),
                str(self.num_parallel), self.activation, str(self.learning_rate), self.optimizer,
                str(self.steps_to_logout),
                ','.join(self.sparse_field_name_list), ','.join(self.dense_field_name_list), str(self.label_size),
                self.need_dropout, str(self.train_data_split)))
        
        #创建上下文管理器，tf.variable_scope，共享大量变量集并且在同一个地方初始化这所有的变量
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
 #==================================================================================================================================           
            

    def load_index_dic(self, feature_field_file):#处理稀疏域，稠密域，以及相应的size
        print(feature_field_file)
        f_field_data = open(feature_field_file)
        sparse_field_name_list = []#稀疏域
        dense_field_name_list = []#稠密域
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
    
    
    
#===========================================================================================================================================================================================
    #感觉是处理输入，经历的过程为    输入层--dense层--embedding层--nn层，输出可以直接进入到网络训练
    def get_user_embedding_list(self, batch_parsed_features, combiner='mean', is_train=True):
        embedding_list = []
        user_long_embedding_size = 0
        for field_name in self.sparse_field_name_list:
            with tf.variable_scope(field_name + "_embedding_layer", reuse=tf.AUTO_REUSE):#；自动实现共享变量 ；embedding层
                field_sparse_ids = batch_parsed_features[field_name]
                field_sparse_values = batch_parsed_features[field_name + "_values"]
                #####组合器
                if combiner == 'mean':
                    embedding = tf.nn.embedding_lookup_sparse(self.tag_embedding_weight, field_sparse_ids,
                                                              field_sparse_values,
                                                              combiner="mean")
                elif combiner == 'avg':
                    embedding = tf.nn.embedding_lookup_sparse(self.tag_embedding_weight, field_sparse_ids,
                                                              field_sparse_values,
                                                              combiner="sum")
                    sparse_features = tf.sparse_merge(field_sparse_ids, field_sparse_values, vocab_size=self.label_size)
                    # tf.sparse_tensor_dense_matmul
                    sparse_x_feature_cnt = tf.sparse_tensor_dense_matmul(sparse_features, self.tag_ones) #（稠密矩阵）self.tag_ones*sparse_features（稀疏矩阵）
                    #tf.div 相除
                    embedding = tf.div(embedding, sparse_x_feature_cnt)  #embedding/（self.tag_ones*sparse_features）
                embedding_list.append(embedding)
                user_long_embedding_size += self.embedding_dim

        for i, field_name in enumerate(self.dense_field_name_list):
            with tf.variable_scope(field_name + "_dense_layer", reuse=tf.AUTO_REUSE):#dense层
                field_dense_feature_values = tf.decode_raw(batch_parsed_features[field_name], tf.float32)#解码函数tf.decode_raw
                embedding_list.append(field_dense_feature_values)
                user_long_embedding_size += self.dense_field_size[i]

        user_long_embedding = tf.concat(embedding_list, 1)
        print("user_long_embedding_size=" + str(user_long_embedding_size))
        
        with tf.variable_scope("user_nn_layer"):#nn层
            ''' layers.get_nn_layer_v2'''
            input_layer_output = layers.get_nn_layer_v2(user_long_embedding, user_long_embedding_size,
                                                        self.nn_layer_shape,
                                                        activation=self.activation,
                                                        l2_reg=self.l2_reg,
                                                need_dropout=(is_train and self.need_dropout))
        '''搞清楚这个函数的意义'''
        return input_layer_output
                          
#===========================================================================imei&&logits======================================================================================================================================================
    
    def test_op(self, batch_parsed_features):
        #调用tf.sparse_to_dense输出一个onehot标签的矩阵
        # labels_one_hot = tf.one_hot(batch_parsed_features["label"],self.label_size, axis=-1)
        # output = tf.sparse_to_dense(batch_parsed_features["label"],[-1,1000])
        # labels = tf.sparse_tensor_to_dense(tf.sparse_merge(batch_parsed_features["label"],batch_parsed_features["label_values"],self.label_size))#转化为稠密张量
        return batch_parsed_features

    def inference_op(self, batch_parsed_features):#公式推断函数
        with tf.name_scope("inference_op"):
            imei = batch_parsed_features['imei']
            user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features,is_train=False)
            #tf.matmul矩阵乘法， tf.transpose转置函数
            logits = tf.matmul(user_nn_layer_output, tf.transpose(self.tag_embedding_weight))#######  输出层xi * embedding的权重
            #tf.nn.bias_add：一个叫bias的向量加到一个叫value的矩阵上，是向量与矩阵的每一行进行相加，得到的结果和value矩阵大小相同。
            logits = tf.nn.bias_add(logits, self.tag_embedding_biases)#  xi * weights + bias
            logits = tf.nn.sigmoid(logits)#转化输出
            return imei, logits

#=========================================================================训练&预测==================================================================
            
        
    def train_op(self, batch_parsed_features):
        with tf.name_scope("train_op"):
            #batch_labels = batch_parsed_features["label"]
            batch_labels = tf.reshape(batch_parsed_features["label"], [-1, 1])
            user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features,is_train=True)
            #计算交叉熵损失
            cross_entropy = tf.nn.sampled_softmax_loss(self.tag_embedding_weight,
                                                       self.tag_embedding_biases,
                                                       batch_labels,
                                                       user_nn_layer_output,
                                                       self.softmax_sampled_cnt,
                                                       self.label_size,
                                                       partition_strategy="div")
            '''
            logits = tf.matmul(user_nn_layer_output, tf.transpose(self.tag_embedding_weight))
            logits = tf.nn.bias_add(logits, self.tag_embedding_biases)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_labels, logits=logits)
            '''
            #交叉熵的均值
            loss = tf.reduce_mean(cross_entropy)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)#使用ada最小化损失
            #tf.summary.scalar用来显示标量信息，一般在画loss,accuary时会用到这个函数
            tf.summary.scalar('loss', loss)

            return train_step, loss, global_step



    def map_op(self, batch_parsed_features):#预测函数
        with tf.name_scope("map_op"):
            batch_labels = tf.sparse_tensor_to_dense(
            tf.sparse_merge(batch_parsed_features["label"], batch_parsed_features["label_values"], self.label_size))
            user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features, is_train=False)
            #tf.matmul，专门矩阵或者tensor乘法
            logits = tf.matmul(user_nn_layer_output, tf.transpose(self.tag_embedding_weight))
            logits = tf.nn.bias_add(logits, self.tag_embedding_biases)
            predictions = tf.nn.sigmoid(logits, name='prediction')
            return predictions,batch_labels
    

    def validate_op(self, batch_parsed_features):#这个函数用来验证？？？
        #batch_labels = tf.reshape(batch_parsed_features["label"], [-1, 1])
        with tf.name_scope("validate_op"):
            batch_labels = batch_parsed_features["label"]
            user_nn_layer_output = self.get_user_embedding_list(batch_parsed_features,is_train=False)
            '''
            cross_entropy = tf.nn.sampled_softmax_loss(self.tag_embedding_weight,
                                                       self.tag_embedding_biases,
                                                       batch_labels,
                                                       user_nn_layer_output,
                                                       self.softmax_sampled_cnt,
                                                       self.label_size,
                                                       partition_strategy="div")
            '''
            logits = tf.matmul(user_nn_layer_output, tf.transpose(self.tag_embedding_weight))
            logits = tf.nn.bias_add(logits, self.tag_embedding_biases)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=batch_labels, logits=logits)
            loss = tf.reduce_mean(cross_entropy)
            return loss

    def fit(self, tf_data_path):
        tf_data_files = input_reader_v2.get_files(tf_data_path)
        #random.shuffle随机排序函数
        random.shuffle(tf_data_files)
        #math.floor() 返回小于或等于一个给定数字的最大整数
        validate_file_num = max(math.floor(len(tf_data_files) * (1 - self.train_data_split)), 1)
        train_files = tf_data_files
        validate_files = input_reader_v2.get_files('./parse_data_tools/data/tf_test_path/20180621')
        #train_files = tf_data_files[:-validate_file_num]
        #validate_files = tf_data_files[-validate_file_num:]
        logging.info("train_files : {}".format(','.join(train_files)))
        logging.info("validate_files : {}".format(','.join(validate_files)))
        
        
        '''input_reader_v2.get_input_v3可能是调用函数：读取作用'''
        next_element = input_reader_v2.get_input_v3(train_files,
                                                 self.dense_field_name_list,
                                                 self.sparse_field_name_list,
                                                 self.num_parallel,
                                                 self.batch_size,
                                                 self.n_epoch,
                                                 buffer_size=self.batch_size * 10)

        train_logit = self.train_op(next_element)

        validate_element = input_reader_v2.get_input(validate_files,
                                                     self.dense_field_name_list,
                                                     self.sparse_field_name_list,
                                                     self.num_parallel,
                                                     self.batch_size * 10,
                                                     self.n_epoch * 10,
                                                     buffer_size=self.batch_size * 10)
        validate_logit = self.map_op(validate_element)

#======================================================流程结构，看不懂===============================================================================
        # tf.group是流程控制函数，该操作可以对 TensorFlow 的多个操作进行分组。 
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        vs = tf.trainable_variables()
        print('There are %d train_able_variables in the Graph: ' % len(vs))
        for v in vs:
            print(v)
        checkpoint_path = "./checkpoint/" + self.method + "/"
        try:
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        except Exception as e:
            logging.info("Fail to remove checkpoint_path, exception: {}".format(e))
        os.makedirs(checkpoint_path)
        checkpoint_file = path.join(checkpoint_path, "checkpoint.ckpt")
        saver = tf.train.Saver(max_to_keep=10)

#===========================================================运行结构部分，写入和储存==================================================================

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
                    train_step, loss, global_step = sess.run(train_logit)
                    loss_sum += loss
                    
                    steps_sum += 1  #迭代步数
                    if global_step % self.steps_to_logout == 0:
                        result = sess.run(merged)
                        writer.add_summary(result, global_step)
                        predictions, batch_labels = sess.run(validate_logit)
                        predictions = predictions[:,1:]
                        batch_labels = batch_labels[:,1:]
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


    def user_inference(self, data_path):
        data_files = input_reader_v2.get_files(data_path)
        next_element = input_reader_v2.get_input(data_files,
                                                 self.dense_field_name_list,
                                                 self.sparse_field_name_list,
                                                 self.num_parallel,
                                                 self.batch_size,
                                                 1,
                                                 buffer_size=self.batch_size * 10)
        saver = tf.train.Saver()
        user_inference_op = self.inference_op(next_element)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess, tf.train.latest_checkpoint("./checkpoint" + self.method))
            try:
                while True:
                    imei, logit = sess.run(user_inference_op)
                    imei = [str(i, encoding="utf-8") for i in imei]
                    logit = [sorted([(i, v) for i, v in enumerate(predicts)], key=lambda x: x[1], reverse=True)[0:500]
                             for predicts in logit]
                    logit = [' '.join([str(i[0]) + ':' + str(i[1]) for i in predicts]) for predicts in logit]
                    output = dict(zip(imei, logit))
                    for imei, predicts in output.items():
                        print(imei + ' ' + predicts)
                    break
            except tf.errors.OutOfRangeError:
                print("End of dataset")
