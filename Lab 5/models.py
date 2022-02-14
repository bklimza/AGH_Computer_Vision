import tensorflow as tf
import numpy as np
from keras.layers import MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate, Input, Dropout, Conv2D
from keras.models import Model

tf.compat.v1.enable_eager_execution()

EPOCHS = 25


def fire_module(x, s1x1, e1x1, e3x3, name):
    # Squeeze layer
    squeeze = Conv2D(s1x1, (1, 1), activation='relu', padding='valid', kernel_initializer='glorot_uniform',
                     name=name + 's1x1')(x)
    squeeze_bn = BatchNormalization(name=name + 'sbn')(squeeze)

    # Expand 1x1 layer and 3x3 layer are parallel

    # Expand 1x1 layer
    expand1x1 = Conv2D(e1x1, (1, 1), activation='relu', padding='valid', kernel_initializer='glorot_uniform',
                       name=name + 'e1x1')(squeeze_bn)

    # Expand 3x3 layer
    expand3x3 = Conv2D(e3x3, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform',
                       name=name + 'e3x3')(squeeze_bn)

    # Concatenate expand1x1 and expand 3x3 at filters
    output = Concatenate(axis=3, name=name)([expand1x1, expand3x3])

    return output


class SqeezeNet:
    def __init__(self, dataset, class_num, batch_size, input_size):
        self.class_num = class_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.dataset = dataset
        self.model = self.__create_model()
        
    def __loss_angle(self, y_true, y_pred, alpha=0.005):
        # cross entropy loss
        bin_true = y_true[:, 0]
        cont_true = y_true[:, 1]
        cls_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf.keras.utils.to_categorical(bin_true,66), logits=y_pred)
        # MSE loss
        pred_cont = tf.reduce_sum(tf.nn.softmax(y_pred) * self.idx_tensor, 1) * 3 - 99
        mse_loss = tf.losses.mean_squared_error(cont_true, pred_cont)
        # Total loss
        total_loss = cls_loss + alpha * mse_loss
        return total_loss

    def __create_model(self):
        inputs = Input(shape=(self.input_size, self.input_size, 3))
        conv1 = Conv2D(96, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', name='Conv1')(inputs)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='Maxpool1')(conv1)
        batch1 = BatchNormalization(name='Batch1')(maxpool1)
        fire4 = fire_module(batch1, 32, 128, 128, "Fire2")
        maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Maxpool2')(fire4)
        fire6 = fire_module(maxpool4, 48, 192, 192, "Fire3")
        fire7 = fire_module(fire6, 48, 192, 192, "Fire4")
        fire8 = fire_module(fire7, 48, 192, 192, "Fire5")
        maxpool8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='Maxpool5')(fire8)
        dropout = Dropout(0.5, name="Dropout")(maxpool8)
        conv10 = Conv2D(10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='Conv6')(dropout)
        batch10 = BatchNormalization(name='Batch6')(conv10)
        avgpool10 = GlobalAveragePooling2D(name='GlobalAvgPool6')(batch10)

        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(avgpool10)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(avgpool10)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(avgpool10)

        model = Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll])
        
        losses = {
            'yaw': self.__loss_angle,
            'pitch': self.__loss_angle,
            'roll': self.__loss_angle}
        
        model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), run_eagerly=True, loss=losses)
        return model

    def train(self, model_path, max_epoches=EPOCHS, load_weight=True):
        self.model.summary()
        
        if load_weight:
            self.model.load_weights(model_path)
        else:
            self.model.fit_generator(generator=self.dataset.data_generator(test=False),
                                    epochs=max_epoches,
                                    steps_per_epoch=self.dataset.train_num // self.batch_size,
                                    max_queue_size=10,
                                    workers=1,
                                    verbose=1)

            self.model.save(model_path)
            
    def test(self, save_dir):
        for i, (images, [batch_yaw, batch_pitch, batch_roll], names) in enumerate(self.dataset.data_generator(test=True)):
            predictions = self.model.predict(images, batch_size=self.batch_size, verbose=1)
            predictions = np.asarray(predictions)
            print(predictions)
            pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, 1)* 3 - 99
            pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, 1)* 3 - 99
            pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, 1)* 3 - 99

            for i in range(len(names)):
                self.dataset.save_test(names[i], save_dir, [pred_cont_yaw[i], pred_cont_pitch[i], pred_cont_roll[i]])
                self.dataset.save_test_real(names[i], save_dir, [batch_yaw[i][0], batch_pitch[i][0], batch_roll[i][0]])


    def test_online(self, image, imageName, save_dir):
        predictions = self.model.predict(image, batch_size=1, verbose=1)
        predictions = np.asarray(predictions)
        pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0, :, :]) * self.idx_tensor, 1) *3 - 180
        pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1, :, :]) * self.idx_tensor, 1) * 3 - 180
        pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2, :, :]) * self.idx_tensor, 1) * 3 - 180

        self.dataset.save_test(imageName, save_dir, [pred_cont_yaw, pred_cont_pitch, pred_cont_roll])
        