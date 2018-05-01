from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, clone_model
from keras.optimizers import Adam
import tensorflow as tf

import matplotlib.pyplot as plt

import sys

import numpy as np

class ADDA():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.src_flag = False
        
        self.optimizer = Adam(0.0002, 0.5)
        '''
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        '''
        
    def define_source_encoder(self, weights=None):
    
        self.source_encoder = Sequential()
        
        inp = Input(shape=self.img_shape)
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.img_shape, padding='same')(inp)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        self.src_flag = True
        
        self.source_encoder = Model(inputs=(inp), outputs=(x))
        
        if weights is not None:
            self.source_encoder.load_weights(weights)
    
    def define_target_encoder(self, weights=None):
        
        if not self.src_flag:
            self.define_source_encoder()
        
        with tf.device('/cpu:0'):
            self.target_encoder = clone_model(self.source_encoder)
            self.target_encoder.set_weights(self.source_encoder.get_weights())
        
        return self.target_encoder
    
    def get_source_classifier(self, model, weights=None):
        
        x = Flatten()(model.output)
        x = Dense(128, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)
        
        source_classifier_model = Model(inputs=(model.input), outputs=(x))
        
        if weights is not None:
            source_classifier_model.load_weights(weights)
        
        return source_classifier_model
    
    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)



if __name__ == '__main__':
    adda = ADDA()
    adda.define_source_encoder()
    model = adda.get_source_classifier(adda.source_encoder)
    model.summary()
    adda.define_target_encoder()
    model = adda.get_source_classifier(adda.target_encoder)
    model.summary()
