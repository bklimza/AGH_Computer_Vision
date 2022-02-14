import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, BatchNormalization
from tensorflow import keras


class Image:
    def __init__(self, img_path, obj_class=0, grayscale=True):      
        self.img_path = img_path
        self.grayscale = grayscale
        self._reset()
        self.color_border = 0.05
        self.obj_class = obj_class

    def _reset(self):
        print('_reset() - img_path={}'.format(self.img_path))   
        if self.grayscale:
            self.im = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE) / 255
        else:
            self.im = cv2.imread(self.img_path)
        self.h = self.im.shape[0]
        self.w = self.im.shape[1]

    def _get_top_boundary(self):
        for row_id in range(self.h):
            lightness_pixel_in_a_row = max(self.im[row_id])
            if lightness_pixel_in_a_row > self.color_border:
                return row_id

    def _get_bottom_boundary(self):
        for row_id in reversed(range(self.h)):
            lightness_pixel_in_a_row = max(self.im[row_id])
            if lightness_pixel_in_a_row > self.color_border:
                return row_id

    def _get_left_boundary(self):
        for row_id in range(self.w):
            lightness_pixel_in_a_row = max(self.im[:, row_id])
            if lightness_pixel_in_a_row > self.color_border:
                return row_id

    def _get_right_boundary(self):
        for row_id in reversed(range(self.w)):
            lightness_pixel_in_a_row = max(self.im[:, row_id])
            if lightness_pixel_in_a_row > self.color_border:
                return row_id

    def get_boundaries(self):
        top_boundary = self._get_top_boundary()
        bottom_boundary = self._get_bottom_boundary()
        left_boundary = self._get_left_boundary()
        right_boundary = self._get_right_boundary()
        return top_boundary, bottom_boundary, left_boundary, right_boundary

    def _get_center(self, top_boundary, bottom_boundary, left_boundary, right_boundary):
        x = (right_boundary / 2 + left_boundary / 2) / self.w
        y = (bottom_boundary / 2 + top_boundary / 2) / self.h
        return x, y

    def _get_width_and_height(self, top_boundary, bottom_boundary, left_boundary, right_boundary):
        width = 1 / (self.w / (right_boundary - left_boundary))
        height = 1 / (self.h / (bottom_boundary - top_boundary))
        return width, height

    def get_date(self):
        top_boundary, bottom_boundary, left_boundary, right_boundary = self.get_boundaries()
        center = self._get_center(top_boundary, bottom_boundary, left_boundary, right_boundary)
        width, height = self._get_width_and_height(top_boundary, bottom_boundary, left_boundary, right_boundary)
        return self.obj_class, center, width, height

    def show(self):
        resized_im = cv2.resize(self.im, (self.w, self.h))
        cv2.imshow("image", resized_im)
        cv2.waitKey()

    def resize(self, w, h):
        self.im = cv2.resize(self.im, (w, h), interpolation=cv2.INTER_AREA)


folder = 'real'
obj = 'duck'  # duck or drill


def load_data(channels, im_size):
    cwd = os.getcwd()
    thumbnails_path = f'{cwd}/datasets/{folder}/{obj}'  # f'{pwd}/dataset'

    print('load_data() - thumbnails_path={}'.format(thumbnails_path))
    list_of_images = os.listdir(thumbnails_path)

    x, y, names = [], [], []

    train_part = .85

    for name in list_of_images:
        if 'jpg' in name:
            image = Image(thumbnails_path + '\\' + name, grayscale=False if channels > 0 else True)
            x.append(image.im)
            names.append(name)

            with open('{}/{}-attributes.txt'.format(thumbnails_path, name[:-9 if "mask" in name else -4]), 'r') as f:
                _ = f.readline()
                _ = f.readline()
                kp_line = f.readline()
                string_values = kp_line.split(',')
                values = [(lambda x: float(x) / im_size)(x) for x in
                          string_values]
                y.append(values)
        if 'png' in name:
            image = Image(thumbnails_path + '/' + name, grayscale=False if channels > 0 else True)
            x.append(image.im)
            names.append(name)

            with open('{}/{}.txt'.format(thumbnails_path, name[:-4]), 'r') as f:
                _ = f.readline()
                _ = f.readline()
                kp_line = f.readline()
                string_values = kp_line.split(',')
                values = [(lambda x: float(x) / im_size)(x) for x in
                          string_values]  # [(lambda x: float(x)/im_size)(x) for x in string_values]
                # values = values[:2]
                y.append(values)

    for _ in range(len(x) * 10):
        i = random.randint(0, len(x) - 1)
        j = random.randint(0, len(x) - 1)
        x[i], x[j] = x[j], x[i]
        y[i], y[j] = y[j], y[i]
        names[i], names[j] = names[j], names[i]

    return np.array(x[:int(len(x) * train_part)]), np.array(y[:int(len(y) * train_part)]), np.array(
        x[int(len(x) * train_part):]), np.array(y[int(len(y) * train_part):]), names


if __name__ == '__main__':
    im_size = 128
    channels = 3
    bs = 24  # 12 24
    ep = 50
    ll = 0.0001
    weights_file = 'kinect-keypoints-{}-{}v2-{}.h5'.format(folder, obj, ep)

    # switch to train/load model
    load = False

    x_train, y_train, x_test, y_test, names = load_data(channels, im_size)

    x_train = x_train.reshape(x_train.shape[0], im_size, im_size, channels)
    x_test = x_test.reshape(x_test.shape[0], im_size, im_size, channels)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model = Sequential()

    model.add(
        Conv2D(64, (5, 1), activation='relu', input_shape=(im_size, im_size, channels), data_format='channels_last'))
    model.add(Conv2D(64, (1, 5), activation='relu'))
    model.add(Conv2D(128, (5, 5), strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 1), activation='relu'))
    model.add(Conv2D(128, (1, 5), activation='relu'))
    model.add(Conv2D(256, (5, 5), strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # model.add(Conv2D(256, (5, 1), activation='relu'))
    # model.add(Conv2D(256, (1, 5), activation='relu'))
    # model.add(Conv2D(256, (5, 5), strides=2, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.25))

    model.add(Conv2D(256, (4, 1), activation='relu'))
    model.add(Conv2D(256, (1, 4), activation='relu'))
    model.add(Conv2D(128, (4, 4), strides=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 1), activation='relu'))
    model.add(Conv2D(128, (1, 3), activation='relu'))
    # model.add(Conv2D(128, (3, 1), activation='relu'))
    # model.add(Conv2D(128, (1, 3), activation='relu'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(16, activation='sigmoid'))

    if not load:
        opt = keras.optimizers.Adam(learning_rate=ll)
        model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy'])

        history = model.fit(x_train, y_train, batch_size=bs, validation_split=.15, epochs=ep, verbose=1)
        print(history.history.keys())
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"], "g")
        plt.show()

        model.save_weights(weights_file)
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Loss, Accuracy: ", score)

        for i in range(9):
            x = x_test[-10 + i: -9 + i]
            y = y_test[-10 + i: -9 + i]
            out = model.predict(x)
            print('gt', y[0])
            print('out', out[0])
            scaled_gt = [(lambda x: int(x * im_size))(x) for x in y[0]]
            scaled_out = [(lambda x: int(x * im_size))(x) for x in out[0]]
            print('gt in px', scaled_gt)
            print('out in px', scaled_out)
            image = Image('./datasets/{}/{}/'.format(folder, obj) + str(names[-10 + i]), grayscale=False)
            for j in range(0, len(scaled_out), 2):
                image.im[scaled_gt[j + 1] - 1, scaled_gt[j] - 1] = (0, 255, 0)
                image.im[scaled_out[j + 1] - 1, scaled_out[j] - 1] = (0, 0, 255)
            cv2.imwrite('results/result_{}_{}.png'.format(obj, i), image.im)

    else:
        # load model and run for random image from dataset
        model.load_weights(weights_file)

        gt = y_train[0:1]
        print(x_train)
        out = model.predict(x_train[0:1])
        print('gt', gt[0])
        print('out', out[0])
        scaled_gt = [(lambda x: int(x * im_size))(x) for x in gt[0]]
        scaled_out = [(lambda x: int(x * im_size))(x) for x in out[0]]
        print('gt in px', scaled_gt)
        print('out in px', scaled_out)
        print(names[0])
        image = Image('./datasets/{}/{}/'.format(folder, obj) + str(names[0]), grayscale=False)
        for i in range(0, len(scaled_out), 2):
            image.im[scaled_gt[i+1], scaled_gt[i]] = (0, 255, 0)
            image.im[scaled_out[i+1], scaled_out[i]] = (0, 0, 255)
        cv2.imwrite('result.png', image.im)
        image.show()
