import VGG16_keras as vgg16
import ferDataPreprocessing as ferDataPre
import ckDataPreProcessing as ckDataPre

###########################################CK数据集#######################################
ck_path = r'F:\Facial-Expression-Recognition.Pytorch-master\CK+48'

ck_train_x,ck_train_y = ckDataPre.get_ck_data(ck_path)

###########################################FER2013数据集##################################
raw_data_train_csv = 'train.csv'
raw_data_test_csv  = 'test.csv'
raw_data_val_csv   = 'val.csv'

fer_train_x,fer_train_y = ferDataPre.raw_data_transform_batch(raw_data_train_csv)
#fer_test_x,fer_test_y = ferDataPre.raw_data_transform_batch(raw_data_test_csv)
#fer_val_x,fer_val_y = ferDataPre.raw_data_transform_batch(raw_data_val_csv)

##########################################执行VGG##########################################
vgg16.vgg_model(ck_train_x,ck_train_y)