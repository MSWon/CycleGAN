# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:08:03 2018

@author: jbk48
"""

import scipy
import tensorflow as tf

from keras.layers import Lambda
from keras.initializers import RandomNormal
from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os


class CycleGAN():
    
    def __init__(self, dataset_name):

        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.conv_init = RandomNormal(0, 0.02) # for convolution kernel (mean: 0, std: 0.02)
        
        self.dataset_name = dataset_name
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols)) 
        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.D_patch = (patch, patch, 1)
        
        optimizer = Adam(0.0002, 0.5)
        ### Build Discriminator ###
        self.D_A = self.Discriminator()   ## A class를 구분하는 Discriminator
        self.D_A.compile(loss='mse', optimizer=optimizer)
        
        self.D_B = self.Discriminator()   ## B class를 구분하는 Discriminator
        self.D_B.compile(loss='mse', optimizer=optimizer)
        
        self.lambda_cycle = 10
        ### Build Generator ###
        self.G_AB = self.Generator() ## A class --> B class  G 함수
        self.G_BA = self.Generator() ## B class --> A class  F 함수
        ## input image
        image_A = Input(shape=self.img_shape)  ## A class 이미지
        image_B = Input(shape=self.img_shape)  ## B class 이미지
        ## Generated image (생성모델의 의해 생성된 이미지)
        fake_B = self.G_AB(image_A) ## B class 가짜 생성 이미지 
        fake_A = self.G_BA(image_B) ## A class 가짜 생성 이미지
        ## Reconstructed image (생성 이미지를 다시 원래 이미지로 복원한 복원 이미지)
        reconstruct_A = self.G_BA(fake_B)
        reconstruct_B = self.G_AB(fake_A)
        ## identity
        identity_A = self.G_BA(image_A)
        identity_B = self.G_AB(image_B)
        
        ## Discriminator의 Weight 학습을 안시킴
        self.D_A.trainable = False
        self.D_B.trainable = False
        
        valid_A = self.D_A(fake_A)  ## 가짜 A 이미지를 Discriminator_A에 넣었을 때  점수  D_A(F(B)) (0~1)
        valid_B = self.D_B(fake_B)  ## 가짜 B 이미지를 Discriminator_B에 넣었을 때  점수  D_B(G(A)) (0~1)
        
        self.combined = Model(inputs = [image_A, image_B],
                              outputs = [valid_A, valid_B,
                                         reconstruct_A, reconstruct_B,
                                         identity_A, identity_B])
    
        self.combined.compile(loss = ["mse", "mse",
                                      "mae", "mae",
                                      "mae", "mae"],
                              loss_weights = [1, 1,
                                              self.lambda_cycle, self.lambda_cycle,
                                              1, 1],
                              optimizer = optimizer)
        
    def Generator(self):
        
        def conv_layer_G(input_data, n_filter, kernel_size=3 , padding = "reflect", instance_norm=True, strides=2):            
            if(padding == "reflect"):               
                x = Lambda(lambda x: tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT'))(input_data)
                x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
                           kernel_initializer=self.conv_init)(x)          
            else:
                x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides, padding = "same",
                           kernel_initializer=self.conv_init)(input_data) 
            x = InstanceNormalization()(x)    
            x = Activation("relu")(x)            
            return x

        def res_block(input_data, n_filter, kernel_size = 3 , instance_norm = True, strides = 1):            
            x = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT'))(input_data) 
            x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
                       kernel_initializer=self.conv_init)(x)          
            x = InstanceNormalization()(x)
            x = Activation("relu")(x)
            
            x = Lambda(lambda x: tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], 'REFLECT'))(x)
            x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
                       kernel_initializer=self.conv_init)(x)          
            x = InstanceNormalization()(x)
            shortcut = input_data  ## x
            merged = Add()([x, shortcut])             
            return merged
        
        def deconv_layer_G(input_data, n_filter, kernel_size = 3 , instance_norm = True, strides = 2, padding = "same"):            
            
            x = Conv2DTranspose(filters=n_filter, kernel_size=kernel_size, strides=strides, padding=padding,
                                kernel_initializer=self.conv_init)(input_data)          
            x = InstanceNormalization()(x)    
            x = Activation("relu")(x)            
            return x        

        input_image = Input(shape=self.img_shape)
        ## Conv layer
        c7s1_32 = conv_layer_G(input_image, 32, kernel_size=7, strides=1)
        d64 = conv_layer_G(c7s1_32, 64, kernel_size=3, padding = "same", strides=2)
        d128 = conv_layer_G(d64, 128, kernel_size=3, padding = "same", strides=2)
        ## resnet_9blocks
        R128 = res_block(d128, 128)
        R128 = res_block(R128, 128)
        R128 = res_block(R128, 128)
        R128 = res_block(R128, 128)
        R128 = res_block(R128, 128)
        R128 = res_block(R128, 128)
        R128 = res_block(R128, 128)
        R128 = res_block(R128, 128)
        R128 = res_block(R128, 128)
        ## deConv layer
        u64 = deconv_layer_G(R128, 64, kernel_size=3, strides=2)  ## shape 버그가 있음
        u32 = deconv_layer_G(u64, 32, kernel_size=3, strides=2)
        u32 = Lambda(lambda x: tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT'))(u32)
        c731_3 = Conv2D(3, kernel_size=7, strides=1, activation='tanh',
                        kernel_initializer=self.conv_init)(u32)  ## c731_3._keras_shape
        output = c731_3
        
        model = Model(inputs = input_image, outputs = output)
        print("## Generator ##")        
        model.summary()
        
        return model
        
    def Discriminator(self):
        
        def conv_layer_D(input_data, n_filter, kernel_size = 4 , instance_norm = True, strides = 2, padding = "same"):            
            
            x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides, padding=padding,
                       kernel_initializer=self.conv_init)(input_data)
            if(instance_norm == True):
                x = InstanceNormalization()(x)  
            x = LeakyReLU(alpha=0.2)(x) 
            return x
        
        input_image = Input(shape=self.img_shape)
        C64 = conv_layer_D(input_image, 64, instance_norm = False)  ## C64
        C128 = conv_layer_D(C64, 128, instance_norm = True)  ## C128
        C256 = conv_layer_D(C128, 256, instance_norm = True)  ## C256
        C512 = conv_layer_D(C256, 512, instance_norm = True)  ## C512
        output = Conv2D(1, kernel_size = 4, strides=1, padding = "same", kernel_initializer=self.conv_init)(C512)
        
        model = Model(inputs = input_image, outputs = output)
        print("## Discriminator ##")
        model.summary()
        
        return model
    
    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        real = np.ones((batch_size,) + self.D_patch)
        fake = np.zeros((batch_size,) + self.D_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Discriminator 학습
                # ----------------------
                
                fake_B = self.G_AB.predict(imgs_A)
                fake_A = self.G_BA.predict(imgs_B)

                # (original images = real / translated = Fake)
                dA_loss_real = self.D_A.train_on_batch(imgs_A, real)
                dA_loss_fake = self.D_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.D_B.train_on_batch(imgs_B, real)
                dB_loss_fake = self.D_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Generator 학습
                # ------------------

                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [real, real,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss,
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
        ## Final image
        for i in range(140):
            self.sample_images(epochs+1, i)
                    

    def sample_images(self, epoch, batch_i):
        
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.G_AB.predict(imgs_A)
        fake_A = self.G_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.G_BA.predict(fake_B)
        reconstr_B = self.G_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

    
    
        