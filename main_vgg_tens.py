import tensorflow as tf
import VGG16_tens as vgg16
import ferDataPreprocessing as ferDataPre
import ckDataPreProcessing as ckDataPre

###########################################CK数据集###################################
ck_path = r'F:\Facial-Expression-Recognition.Pytorch-master\CK+48'

ck_train_x,ck_train_y = ckDataPre.get_ck_data(ck_path)

###########################################FER2013数据集###################################
raw_data_train_csv = 'train.csv'
raw_data_test_csv  = 'test.csv'
raw_data_val_csv   = 'val.csv'

fer_train_x,fer_train_y = ferDataPre.raw_data_transform_batch(raw_data_train_csv)
#Test_x,Test_y         = ferDataPre.raw_data_transform_batch(raw_data_test_csv)
#Val_x,Val_y           = ferDataPre.raw_data_transform_batch(raw_data_val_csv)

##########################################构造计算图#########################################
N_CLASSES  = 7
LR         = 0.0001
EPOCH      = 1000
BATCH_SIZE = 32

x_batch = tf.placeholder(dtype=tf.float32,shape=[None,48,48,3])
y_batch = tf.placeholder(dtype=tf.float32,shape=[None,N_CLASSES])

vgg         = vgg16.VGG16()
y_pred      = vgg.forward(x_batch,N_CLASSES)
loss_op     = vgg.loss(y_pred,y_batch)
train_op    = vgg.training(loss_op,LR)
accuracy_op = vgg.get_accuracy(y_pred,y_batch)

init_op = tf.global_variables_initializer()

########################################构造计算图#####################################
with tf.Session() as sess:
    sess.run(init_op)
    for epoch in range(EPOCH):
        x_train,y_train = ferDataPre.train_data_shuffle_batch(ck_train_x,ck_train_y,BATCH_SIZE)
        train_op.run(feed_dict={x_batch:x_train,y_batch:y_train})
        if epoch % 1 == 0:
            train_accuracy = accuracy_op.eval(feed_dict={x_batch:x_train,y_batch:y_train})
            print("epoch %d, training accuracy %g"%(epoch, train_accuracy))

