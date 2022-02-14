import tensorflow as tf
import datasets
import models
tf.compat.v1.enable_eager_execution()

object_name = "drill"

PROJECT_DIR = "./"


VISAPP_DATA_DIR = 'dataDir/' + object_name
VISAPP_MODEL_FILE = PROJECT_DIR + 'model/' + object_name + '_model.h5'
VISAPP_TEST_SAVE_DIR = 'test/' + object_name


BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 300

dataset = datasets.Visapp(VISAPP_DATA_DIR, 'filename_list.txt', object_name, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, ratio=0.7)

net = models.SqeezeNet(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

net.train(VISAPP_MODEL_FILE, max_epoches=EPOCHS, load_weight=False)

net.test(VISAPP_TEST_SAVE_DIR)
