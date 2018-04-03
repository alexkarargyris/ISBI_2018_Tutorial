
from keras.models import Model
from keras.layers import Input,  concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization


def generate_unet(base_num_filters, num_classes, kernel_size=(3,3), image_width=128, image_height=128):
        """
        Simple UNet without batch normalization
        """
        inputs = Input((image_height,  image_width, 1))
        conv1 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(2*base_num_filters, kernel_size, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(2*base_num_filters, kernel_size, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(4*base_num_filters, kernel_size, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(4*base_num_filters, kernel_size, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(8*base_num_filters, kernel_size, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(8*base_num_filters, kernel_size, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(16*base_num_filters, kernel_size, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(16*base_num_filters, kernel_size, activation='relu', padding='same')(conv5)
        drop = Dropout(0.5)(conv5)

        up6 = concatenate([Conv2DTranspose(8*base_num_filters, kernel_size, strides=(2, 2), padding='same')(drop), conv4], axis=3)
        conv6 = Conv2D(8*base_num_filters, kernel_size, activation='relu', padding='same')(up6)
        conv6 = Conv2D(8*base_num_filters, kernel_size, activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(4*base_num_filters, kernel_size, strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(4*base_num_filters, kernel_size, activation='relu', padding='same')(up7)
        conv7 = Conv2D(4*base_num_filters, kernel_size, activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(2*base_num_filters, kernel_size, strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(2*base_num_filters, kernel_size, activation='relu', padding='same')(up8)
        conv8 = Conv2D(2*base_num_filters, kernel_size, activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(base_num_filters, kernel_size, strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(up9)
        conv9 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(conv9)

        conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        return model



def generate_batch_norm_unet(base_num_filters, num_classes, kernel_size=(3,3), image_width=128, image_height=128):
        """
        UNet with batch normalization
        """
        inputs = Input((image_height, image_width, 1))
        conv1 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(inputs)
        bn1 = BatchNormalization()(conv1)
        conv1 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(bn1)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

        conv2 = Conv2D(2 * base_num_filters, kernel_size, activation='relu', padding='same')(pool1)
        bn2 = BatchNormalization()(conv2)
        conv2 = Conv2D(2 * base_num_filters, kernel_size, activation='relu', padding='same')(bn2)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

        conv3 = Conv2D(4 * base_num_filters, kernel_size, activation='relu', padding='same')(pool2)
        bn3 = BatchNormalization()(conv3)
        conv3 = Conv2D(4 * base_num_filters, kernel_size, activation='relu', padding='same')(bn3)
        bn3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

        conv4 = Conv2D(8 * base_num_filters, kernel_size, activation='relu', padding='same')(pool3)
        bn4 = BatchNormalization()(conv4)
        conv4 = Conv2D(8 * base_num_filters, kernel_size, activation='relu', padding='same')(bn4)
        bn4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

        conv5 = Conv2D(16 * base_num_filters, kernel_size, activation='relu', padding='same')(pool4)
        bn5 = BatchNormalization()(conv5)
        conv5 = Conv2D(16 * base_num_filters, kernel_size, activation='relu', padding='same')(bn5)
        bn5 = BatchNormalization()(conv5)
        drop = Dropout(0.5)(bn5)

        up6 = concatenate([Conv2DTranspose(8*base_num_filters, kernel_size, strides=(2, 2), padding='same')(drop), conv4], axis=3)
        conv6 = Conv2D(8 * base_num_filters, kernel_size, activation='relu', padding='same')(up6)
        bn6 = BatchNormalization()(conv6)
        conv6 = Conv2D(8 * base_num_filters, kernel_size, activation='relu', padding='same')(bn6)
        bn6 = BatchNormalization()(conv6)

        up7 = concatenate([Conv2DTranspose(4*base_num_filters, kernel_size, strides=(2, 2), padding='same')(bn6), conv3], axis=3)
        conv7 = Conv2D(4 * base_num_filters, kernel_size, activation='relu', padding='same')(up7)
        bn7 = BatchNormalization()(conv7)
        conv7 = Conv2D(4 * base_num_filters, kernel_size, activation='relu', padding='same')(bn7)
        bn7 = BatchNormalization()(conv7)

        up8 = concatenate([Conv2DTranspose(2*base_num_filters, kernel_size, strides=(2, 2), padding='same')(bn7), conv2], axis=3)
        conv8 = Conv2D(2 * base_num_filters, kernel_size, activation='relu', padding='same')(up8)
        bn8 = BatchNormalization()(conv8)
        conv8 = Conv2D(2 * base_num_filters, kernel_size, activation='relu', padding='same')(bn8)
        bn8 = BatchNormalization()(conv8)

        up9 = concatenate([Conv2DTranspose(base_num_filters, kernel_size, strides=(2, 2), padding='same')(bn8), conv1], axis=3)
        conv9 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(up9)
        bn9 = BatchNormalization()(conv9)
        conv9 = Conv2D(base_num_filters, kernel_size, activation='relu', padding='same')(bn9)
        bn9 = BatchNormalization()(conv9)

        conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(bn9)

        model = Model(inputs=[inputs], outputs=[conv10])

        return model
