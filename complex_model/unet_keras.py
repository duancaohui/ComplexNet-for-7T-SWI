import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Lambda, MaxPooling2D, Concatenate, UpSampling2D, Add, BatchNormalization, Multiply, concatenate

initializer = tf.random_normal_initializer(0., 0.02)
regularizer = keras.regularizers.l2(0)


def conv2d(filters, size, apply_batchnorm=True):
    result = keras.Sequential()
    result.add(Conv2D(filters, size, strides=1, padding='same',
                      kernel_initializer='he_normal', use_bias=True))

    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization())

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


def downsample(filters, size, apply_batchnorm=True):
    result = keras.Sequential()
    result.add(
        keras.layers.Conv2D(filters, size, strides=2,
                            padding='same',
                            kernel_initializer=initializer,
                            use_bias=False))
    if apply_batchnorm:
        result.add(keras.layers.BatchNormalization())
    result.add(keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    result = keras.Sequential()
    result.add(
        keras.layers.Conv2DTranspose(filters, size, strides=2,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     use_bias=False))
    result.add(keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(keras.layers.Dropout(0.5))
    result.add(keras.layers.ReLU())
    return result


def Fourier(x):
    x_complex = tf.complex(x[..., 0], tf.zeros_like(x)[...,0])
    y_complex = tf.signal.fft2d(x_complex)
    return y_complex


def Fourier2(x):
    x_complex = tf.complex(x[..., 0], x[..., 1])
    y_complex = tf.signal.fft2d(x_complex)
    return y_complex


def Fourier5(x):
    x_complex = tf.complex(x, tf.zeros_like(x))
    x_complex = tf.transpose(x_complex, [0, 3, 1, 2])
    y_complex = tf.signal.fft2d(x_complex)
    y_complex = tf.transpose(y_complex, [0, 2, 3, 1])
    return y_complex


def iFourier5(k):
    corrected_kspace = tf.transpose(k, [0, 3, 1, 2])
    corrected_complex = tf.signal.ifft2d(corrected_kspace)
    corrected_complex = tf.transpose(corrected_complex, [0, 2, 3, 1])
    corrected_mag = tf.abs(corrected_complex)
    return corrected_mag


def Fourier10(x):
    x_complex = tf.complex(x[..., :5], x[..., 5:])
    x_complex = tf.transpose(x_complex, [0, 3, 1, 2])
    y_complex = tf.signal.fft2d(x_complex)
    y_complex = tf.transpose(y_complex, [0, 2, 3, 1])
    return y_complex


def iFourier10(k):
    corrected_kspace = tf.transpose(k, [0, 3, 1, 2])
    corrected_complex = tf.signal.ifft2d(corrected_kspace)
    corrected_complex = tf.transpose(corrected_complex, [0, 2, 3, 1])

    corrected_real = tf.math.real(corrected_complex)
    corrected_imag = tf.math.imag(corrected_complex)
    corrected_real_concat = Concatenate()([corrected_real, corrected_imag])  # batchsize*64*64*2
    return corrected_real_concat


def expand_dims(x):
    return tf.expand_dims(x, -1)


def add_dc_layer(x, features, mask):
    # add dc connection for each block

    first_layer = features
    feature_kspace = Lambda(Fourier)(first_layer)
    projected_kspace = Multiply()([feature_kspace, mask])

    # get output and input
    last_layer = x
    gene_kspace = Lambda(Fourier)(last_layer)
    gene_kspace = Multiply()([gene_kspace, (1.0 - mask)])

    corrected_kspace = Add()([projected_kspace, gene_kspace])

    # inverse fft
    corrected_complex = Lambda(tf.signal.ifft2d)(corrected_kspace)
    corrected_mag = Lambda(tf.abs)(corrected_complex)
    corrected_mag = Lambda(expand_dims)(corrected_mag)

    return corrected_mag


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
    corrected_complex = Lambda(tf.signal.ifft2d)(corrected_kspace)
    corrected_real = Lambda(tf.math.real)(corrected_complex)
    corrected_imag = Lambda(tf.math.imag)(corrected_complex)
    corrected_real = Lambda(expand_dims)(corrected_real)
    corrected_imag = Lambda(expand_dims)(corrected_imag)
    corrected_real_concat = Concatenate()([corrected_real, corrected_imag])  # batchsize*64*64*2

    return corrected_real_concat


def add_dc_layer10(x, features, mask):
    # add dc connection for each block

    first_layer = features
    feature_kspace = Lambda(Fourier10)(first_layer)
    projected_kspace = Multiply()([feature_kspace, mask])

    # get output and input
    last_layer = x
    gene_kspace = Lambda(Fourier10)(last_layer)
    gene_kspace = Multiply()([gene_kspace, (1.0 - mask)])

    corrected_kspace = Add()([projected_kspace, gene_kspace])
    corrected_real_concat = Lambda(iFourier10)(corrected_kspace)
    return corrected_real_concat


def add_dc_layer5(x, features, mask):
    # add dc connection for each block

    first_layer = features
    feature_kspace = Lambda(Fourier5)(first_layer)
    projected_kspace = Multiply()([feature_kspace, mask])

    # get output and input
    last_layer = x
    gene_kspace = Lambda(Fourier5)(last_layer)
    gene_kspace = Multiply()([gene_kspace, (1.0 - mask)])

    corrected_kspace = Add()([projected_kspace, gene_kspace])
    corrected_real_concat = Lambda(iFourier5)(corrected_kspace)
    return corrected_real_concat


def unet(img_width=64, img_height=64, channels=2):

    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    mask = Input(shape=[img_width, img_height, 5], name='undersamling_mask', dtype=tf.complex64)

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

    gene_output = add_dc_layer10(res, inputs, mask)
    model = keras.Model(inputs=[inputs, mask], outputs=[gene_output, res], name='unet')
    return model


def unet5(img_width=64, img_height=64, channels=2):

    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    mask = Input(shape=[img_width, img_height, 5], name='undersamling_mask', dtype=tf.complex64)
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
    gene_output = add_dc_layer10(res, inputs, mask)
    model = keras.Model(inputs=[inputs, mask], outputs=[res, res], name='unet5')
    # model = keras.Model(inputs=[inputs, mask], outputs=gene_output, name='unet5')
    return model


def unet2(img_width=64, img_height=64, channels=2):

    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    # inputs_c = Input(shape=[img_width, img_height, 2], name='zero-fillingc')
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

    up9 = upconv2d(64,3)(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv2d(64, 3)(merge9)
    conv9 = conv2d(64, 3)(conv9)

    outputs = Conv2D(2, 1, strides=1, kernel_initializer='he_normal', use_bias=True)(conv9)
    res = Add(name='res_output')([outputs, inputs])

    gene_output = add_dc_layer2(res, inputs, mask)
    model = keras.Model(inputs=[inputs, mask], outputs=[res, gene_output], name='unet')
    return model


def pix2pix(img_width=64, img_height=64, channels=2):

    inputs = Input(shape=[img_width, img_height, channels], name='zero-filling')  # 96*96*1
    # mask = Input(shape=[IMG_WIDTH, IMG_HEIGHT, 1], name='undersamling_mask')

    conv1 = conv2d(64, 3)(inputs)  # 96*96*64
    conv1 = conv2d(64, 4)(conv1)
    conv1 = conv2d(64, 3)(conv1)
    pool1 = downsample(64, 4)(conv1)

    conv2 = conv2d(128, 3)(pool1)  # 48*48*128
    conv2 = conv2d(128, 3)(conv2)
    pool2 = downsample(128, 4)(conv2)

    conv3 = conv2d(256, 3)(pool2)  # 24*24*256
    conv3 = conv2d(256, 3)(conv3)
    pool3 = downsample(256, 4)(conv3)

    conv4 = conv2d(512, 3)(pool3)  # 12*12*512
    conv4 = conv2d(512, 3)(conv4)
    pool4 = downsample(512, 4)(conv4)

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

    up9 = upconv2d(64,3)(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv2d(64, 3)(merge9)
    conv9 = conv2d(64, 3)(conv9)

    outputs = Conv2D(1, 1, strides=1, kernel_initializer='he_normal', use_bias=True)(conv9)
    res = Add(name='res_output')([outputs, inputs])
    model = keras.Model(inputs=inputs, outputs=res, name='unet')
    return model


def generator_loss(gene_output, features, labels, masks):

    gene_l1l2_factor = 0.5

    # fourier_transform
    gene_kspace = Lambda(Fourier)(gene_output)
    feature_kspace = Lambda(Fourier)(features)
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
    gene_loss = tf.add(gene_dc_loss*0.0001, gene_mse_loss, name='gene_loss')
    return gene_loss


def generator_loss1(gene_output, features, features_c, labels, masks):
    gene_l1l2_factor = 0.5

    # fourier_transform
    gene_kspace = Lambda(Fourier2)(gene_output)
    feature_kspace = Lambda(Fourier2)(features_c)
    feature_mask = masks
    loss_kspace = tf.cast(tf.square(tf.abs(gene_kspace - feature_kspace)), tf.float32) * tf.cast(feature_mask,
                                                                                                 tf.float32)

    # data consistency
    gene_dc_loss = tf.reduce_mean(loss_kspace, name='gene_dc_loss')

    # mse loss
    gene_complex = tf.complex(gene_output[..., 0], gene_output[..., 1])
    gene_mag = tf.abs(gene_complex)
    gene_mag = tf.expand_dims(gene_mag, -1)

    gene_l1_loss = tf.reduce_mean(tf.abs(gene_mag - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_mag - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')

    return gene_mse_loss


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

    # mse loss
    gene_complex = tf.complex(gene_output[..., 0], gene_output[..., 1])
    gene_mag = tf.abs(gene_complex)
    gene_mag = tf.expand_dims(gene_mag, -1)

    gene_l1_loss = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_output - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')

    return gene_loss


def generator_loss2_m(gene_output, features, labels, masks):
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
    gene_complex = tf.complex(gene_output[..., 0], gene_output[..., 1])
    gene_mag = tf.abs(gene_complex)
    gene_mag = tf.expand_dims(gene_mag, -1)

    gene_l1_loss = tf.reduce_mean(tf.abs(gene_mag - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_mag - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')

    return gene_loss


def generator_loss5(gene_output, features, labels, masks):

    gene_l1l2_factor = 0.5

    # fourier_transform
    gene_kspace = Lambda(Fourier5)(gene_output)
    feature_kspace = Lambda(Fourier5)(features)
    feature_mask = masks
    loss_kspace = tf.cast(tf.square(tf.abs(gene_kspace - feature_kspace)), tf.float32) * tf.cast(feature_mask,
                                                                                                 tf.float32)

    # data consistency
    gene_dc_loss = tf.reduce_mean(loss_kspace, name='gene_dc_loss')

    # mse loss
    gene_mag = gene_output
    gene_l1_loss = tf.reduce_mean(tf.abs(gene_mag - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_mag - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')

    return gene_mse_loss


def generator_loss55(gene_output, features, labels, masks):

    gene_l1l2_factor = 0.5

    # fourier_transform
    gene_kspace = Lambda(Fourier10)(gene_output)
    feature_kspace = Lambda(Fourier10)(features)
    feature_mask = masks
    loss_kspace = tf.cast(tf.square(tf.abs(gene_kspace - feature_kspace)), tf.float32) * tf.cast(feature_mask,
                                                                                                 tf.float32)

    # data consistency
    gene_dc_loss = tf.reduce_mean(loss_kspace, name='gene_dc_loss')

    # mse loss
    gene_complex = tf.complex(gene_output[..., :5], gene_output[..., 5:])
    gene_mag = tf.abs(gene_complex)

    gene_l1_loss = tf.reduce_mean(tf.abs(gene_mag - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_mag - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')

    return gene_mse_loss


def generator_loss10(gene_output, features, labels, masks):

    gene_l1l2_factor = 0.5

    # fourier_transform
    gene_kspace = Lambda(Fourier10)(gene_output)
    feature_kspace = Lambda(Fourier10)(features)
    feature_mask = masks
    loss_kspace = tf.cast(tf.square(tf.abs(gene_kspace - feature_kspace)), tf.float32) * tf.cast(feature_mask,
                                                                                                 tf.float32)

    # data consistency
    gene_dc_loss = tf.reduce_mean(loss_kspace, name='gene_dc_loss')

    # mse loss
    gene_complex = tf.complex(gene_output[..., :5], gene_output[..., 5:])
    gene_mag = tf.abs(gene_complex)

    gene_l1_loss = tf.reduce_mean(tf.abs(gene_mag - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_mag - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(gene_dc_loss, gene_mse_loss, name='gene_loss')

    return gene_loss


def generator_loss10_10(gene_output, features, labels, masks):

    gene_l1l2_factor = 0.5

    # fourier_transform
    gene_kspace = Lambda(Fourier10)(gene_output)
    feature_kspace = Lambda(Fourier10)(features)
    feature_mask = masks
    loss_kspace = tf.cast(tf.square(tf.abs(gene_kspace - feature_kspace)), tf.float32) * tf.cast(feature_mask,
                                                                                                 tf.float32)

    # data consistency
    gene_dc_loss = tf.reduce_mean(loss_kspace, name='gene_dc_loss')

    # mse loss
    # gene_complex = tf.complex(gene_output[..., :5], gene_output[..., 5:])
    # gene_mag = tf.abs(gene_complex)

    gene_l1_loss = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')
    gene_l2_loss = tf.reduce_mean(tf.square(gene_output - labels), name='gene_l2_loss')
    gene_mse_loss = tf.add(gene_l1l2_factor * gene_l1_loss,
                           (1.0 - gene_l1l2_factor) * gene_l2_loss, name='gene_mse_loss')

    # gene_mse_factor as a parameter
    gene_loss = tf.add(0.01*gene_dc_loss, gene_mse_loss, name='gene_loss')

    return gene_loss