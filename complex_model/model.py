import tensorflow as tf
from tensorflow import keras
from complex_model.utils import ComplexInit
from tensorflow.keras.layers import Input, Lambda, Add, Multiply, Conv2DTranspose, MaxPooling2D, concatenate, Conv2D


class ComplexConv2D(keras.layers.Layer):

    def __init__(self, kw, kh, n_out, sw, sh, activation):
        super(ComplexConv2D, self).__init__()
        self.kw = kw
        self.kh = kh
        self.n_out = n_out
        self.sw = sw
        self.sh = sh
        self.activation = activation

    def build(self, input_shape):
        n_in = input_shape[-1] // 2
        kernel_init = ComplexInit(kernel_size=(self.kh, self.kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=self.n_out,
                                  criterion='he')

        self.w = self.add_weight(name='w',
                                 shape=(self.kh, self.kw, n_in, self.n_out*2),
                                 initializer=kernel_init,
                                 trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(self.n_out*2,),
                                 initializer=keras.initializers.Constant(0.0001),
                                 trainable=True)

    def call(self, inputs):
        kernel_real = self.w[:, :, :, :self.n_out]
        kernel_imag = self.w[:, :, :, self.n_out:]
        cat_kernel_real = tf.concat([kernel_real, -kernel_imag], axis=-2)
        cat_kernel_imag = tf.concat([kernel_imag, kernel_real], axis=-2)
        cat_kernel_complex = tf.concat([cat_kernel_real, cat_kernel_imag], axis=-1)
        conv = tf.nn.conv2d(inputs, cat_kernel_complex, strides=[1, self.sh, self.sw, 1], padding='SAME')
        conv_bias = tf.nn.bias_add(conv, self.b)
        if self.activation:
            act = tf.nn.relu(conv_bias)
            output = act
        else:
            output = conv_bias
        return output

    def get_config(self):
        config = {
            'kw': self.kw,
            'kh': self.kh,
            'n_out': self.n_out,
            'sw': self.sw,
            'sh': self.sh,
            'activation': self.activation
        }
        base_config = super(ComplexConv2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def conv2d(filters, size):
    result = keras.Sequential()
    result.add(keras.layers.Conv2D(filters, size, strides=1, padding='same',
                                   kernel_initializer='he_normal', use_bias=True))

    result.add(keras.layers.ReLU())

    return result


def upconv2d(filters, size, apply_batchnorm=True):
    result = keras.Sequential()
    result.add(Conv2DTranspose(filters, size, strides=2, padding='same',
                               kernel_initializer='he_normal', use_bias=True))

    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization())

    result.add(keras.layers.ReLU())

    return result


def Fourier2(x):
    x_complex = tf.complex(x[..., 0], x[..., 1])
    y_complex = tf.signal.fft2d(x_complex)
    return y_complex


def iFourier2(x):
    corrected_complex = tf.signal.ifft2d(x)
    corrected_real = tf.math.real(corrected_complex)
    corrected_imag = tf.math.imag(corrected_complex)
    y_complex = tf.stack([corrected_real, corrected_imag], axis=-1)
    return y_complex


def add_dc_layer2(x, features, mask):
    # add dc connection for each block
    first_layer = features
    feature_kspace = Lambda(Fourier2)(first_layer)
    projected_kspace = Multiply()([feature_kspace, mask])

    # get output and input
    last_layer = x
    gene_kspace = Lambda(Fourier2)(last_layer)
    gene_kspace = Multiply()([gene_kspace, (1.0 - mask)])

    corrected_kspace = Add()([projected_kspace, gene_kspace])

    # inverse fft
    corrected_real_concat = Lambda(iFourier2)(corrected_kspace)

    return corrected_real_concat


def getModel(img_width, img_height, channels, fea_num):
    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    mask = Input(shape=[img_width, img_height], name='undersamling_mask', dtype=tf.complex64)
    temp = inputs
    # fea_num = 128
    for i in range(5):
        conv1 = ComplexConv2D(kw=3, kh=3, n_out=fea_num, sw=1, sh=1, activation=True)(temp)
        conv2 = ComplexConv2D(kw=3, kh=3, n_out=fea_num, sw=1, sh=1, activation=True)(conv1)
        conv3 = ComplexConv2D(kw=3, kh=3, n_out=fea_num, sw=1, sh=1, activation=True)(conv2)
        conv4 = ComplexConv2D(kw=3, kh=3, n_out=fea_num, sw=1, sh=1, activation=True)(conv3)
        conv5 = ComplexConv2D(kw=3, kh=3, n_out=1, sw=1, sh=1, activation=False)(conv4)
        block = Add()([conv5, temp])
        temp = add_dc_layer2(block, inputs, mask)

    model = keras.Model(inputs=[inputs, mask], outputs=temp, name='deep_complex')
    return model


def unet(img_width=64, img_height=64, channels=2):

    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    mask = Input(shape=[img_width, img_height], name='undersamling_mask', dtype=tf.complex64)

    conv1 = conv2d(64, 3)(inputs)  # 96*96*64
    conv1 = conv2d(64, 4)(conv1)
    conv1 = conv2d(64, 3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2d(128, 3)(pool1)  # 48*48*128
    conv2 = conv2d(128, 3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2d(256, 3)(pool2)  # 24*24*256
    conv3 = conv2d(256, 3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv2d(512, 3)(pool3)  # 12*12*512
    conv4 = conv2d(512, 3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv2d(1024, 3)(pool4)  # 6*6*1024
    conv5 = conv2d(1024, 3)(conv5)

    up6 = upconv2d(512, 3)(conv5)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = conv2d(512, 3)(merge6)
    conv6 = conv2d(512, 3)(conv6)

    up7 = upconv2d(256, 3)(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = conv2d(256, 3)(merge7)
    conv7 = conv2d(256, 3)(conv7)

    up8 = upconv2d(256, 3)(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = conv2d(128, 3)(merge8)
    conv8 = conv2d(128, 3)(conv8)

    up9 = upconv2d(64, 3)(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv2d(64, 3)(merge9)
    conv9 = conv2d(64, 3)(conv9)

    outputs = Conv2D(channels, 1, strides=1, kernel_initializer='he_normal', use_bias=True)(conv9)

    res = Add(name='res_output')([outputs, inputs])

    gene_output = add_dc_layer2(res, inputs, mask)
    model = keras.Model(inputs=[inputs, mask], outputs=gene_output, name='unet')
    return model


def unet_stage4(img_width=64, img_height=64, channels=2):

    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    mask = Input(shape=[img_width, img_height], name='undersamling_mask', dtype=tf.complex64)

    conv1 = conv2d(64, 3)(inputs)  # 96*96*64
    conv1 = conv2d(64, 4)(conv1)
    conv1 = conv2d(64, 3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2d(128, 3)(pool1)  # 48*48*128
    conv2 = conv2d(128, 3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2d(256, 3)(pool2)  # 24*24*256
    conv3 = conv2d(256, 3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv2d(512, 3)(pool3)  # 12*12*512
    conv4 = conv2d(512, 3)(conv4)


    up5 = upconv2d(512, 3)(conv4)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = conv2d(512, 3)(merge5)
    conv5 = conv2d(512, 3)(conv5)

    up6 = upconv2d(256, 3)(conv5)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = conv2d(256, 3)(merge6)
    conv6 = conv2d(256, 3)(conv6)

    up7 = upconv2d(256, 3)(conv6)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = conv2d(128, 3)(merge7)
    conv7 = conv2d(128, 3)(conv7)

    outputs = Conv2D(channels, 1, strides=1, kernel_initializer='he_normal', use_bias=True)(conv7)

    res = Add(name='res_output')([outputs, inputs])

    gene_output = add_dc_layer2(res, inputs, mask)
    model = keras.Model(inputs=[inputs, mask], outputs=gene_output, name='unet')
    return model


def realResNet(img_width, img_height, channels, fea_num):
    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    mask = Input(shape=[img_width, img_height], name='undersamling_mask', dtype=tf.complex64)
    temp = inputs
    for i in range(5):
        conv1 = conv2d(fea_num, 3)(temp)
        conv2 = conv2d(fea_num, 3)(conv1)
        conv3 = conv2d(fea_num, 3)(conv2)
        conv4 = conv2d(fea_num, 3)(conv3)
        conv5 = keras.layers.Conv2D(channels, 1, strides=1, kernel_initializer='he_normal', use_bias=True)(conv4)
        block = Add()([conv5, temp])
        temp = add_dc_layer2(block, inputs, mask)

    model = keras.Model(inputs=[inputs, mask], outputs=temp, name='real_ResNet')
    return model


def realResNet_1cha(img_width, img_height, channels, fea_num):
    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    mask = Input(shape=[img_width, img_height], name='undersamling_mask', dtype=tf.complex64)
    temp = inputs
    for i in range(5):
        conv1 = conv2d(fea_num, 3)(temp)
        conv2 = conv2d(fea_num, 3)(conv1)
        conv3 = conv2d(fea_num, 3)(conv2)
        conv4 = conv2d(fea_num, 3)(conv3)
        conv5 = keras.layers.Conv2D(channels, 1, strides=1, kernel_initializer='he_normal', use_bias=True)(conv4)
        block = Add()([conv5, temp])
        temp = block

    model = keras.Model(inputs=[inputs, mask], outputs=temp, name='real_ResNet')
    return model


def generator_loss2(gene_output, features, labels, masks):

    gene_l1l2_factor = 0.5

    # fourier_transform
    gene_kspace = Lambda(Fourier2)(gene_output)
    feature_kspace = Lambda(Fourier2)(features)
    feature_mask = masks
    loss_kspace = tf.cast(tf.square(tf.abs(gene_kspace - feature_kspace)), tf.float32) * tf.cast(feature_mask,
                                                                                                 tf.float32)

    # data consistency
    gene_dc_loss = tf.reduce_mean(loss_kspace, name='gene_dc_loss')

    gene_complex = tf.complex(gene_output[..., 0], gene_output[..., 1])
    gene_mag = tf.abs(gene_complex)
    gene_mag = tf.expand_dims(gene_mag, -1)

    # mse loss
    gene_l1_loss = tf.reduce_mean(tf.abs(gene_mag - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_mag - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')
    return gene_loss


def generator_loss(gene_output, features, labels, masks):

    gene_l1l2_factor = 0.5

    # fourier_transform
    gene_kspace = Lambda(Fourier2)(gene_output)
    feature_kspace = Lambda(Fourier2)(features)
    feature_mask = masks
    loss_kspace = tf.cast(tf.square(tf.abs(gene_kspace - feature_kspace)), tf.float32) * tf.cast(feature_mask,
                                                                                                 tf.float32)

    # data consistency
    gene_dc_loss = tf.reduce_mean(loss_kspace, name='gene_dc_loss')

    # mse loss
    gene_l1_loss = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_output - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')
    return gene_loss


def generator_loss_1cha(gene_output, features, labels, masks):

    gene_l1l2_factor = 0.5

    # mse loss
    gene_l1_loss = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_output - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')
    return gene_mse_loss