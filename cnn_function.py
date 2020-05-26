import tensorflow as tf
import numpy as np
import scipy.misc as scp
from math import ceil
from tensorflow.python.training import moving_averages

def plot_filters(data, padsize=1, padval=0):
    data=data.transpose(3,0,1,2)
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return(data)


def conv_layer_init(name,x,shape,w_init="xavier",b_init="constant",mean=0.0, stddev=1, b_value=0.1,stride=[1,1],
padding='SAME',activation=True,batch_norm=False,training=False):
    with tf.variable_scope(name) as scope:
        if w_init=="xavier":
            weights = tf.get_variable("weights",shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        elif w_init=="constant":
            weights = tf.get_variable("weights",shape,initializer=tf.constant_initializer(0))
        elif w_init=="normal":
            weights = tf.get_variable("weights",initializer=tf.random_normal(shape,mean=mean,stddev=stddev))
        else:
            raise ValueError("The  initialization "+ w_init+ " is invalid for the weights of the layer "+name)

        if b_init=="constant":
            biases = tf.get_variable("biases", [shape[3]], initializer=tf.constant_initializer(b_value))
        elif b_init=="xavier":
            biases= tf.get_variable("biases", [shape[3]], initializer=tf.contrib.layers.xavier_initializer())
        else:
            raise ValueError("The  initialization "+ b_init+ " is invalid for the bias of the layer "+name)

        conv=tf.nn.conv2d(x, weights, [1, stride[0], stride[1], 1],padding=padding)
        conv=tf.nn.bias_add(conv, biases)
        if batch_norm:
            conv=tf.contrib.layers.batch_norm(conv,activation_fn=None,is_training=training,scope=scope,reuse=False)

        if activation:
            activ =tf.nn.relu(conv)
        else:
            activ=conv
    return activ

def conv3d_layer_init(name,x,shape,w_init="xavier",b_init="xavier",mean=0.0, stddev=1, b_value=0.1,stride=[1,1,1],
padding='SAME',activation=True,batch_norm=False,training=False):
#input: A Tensor. Must be one of the following types: float32, float64. Shape [batch, in_depth, in_height, in_width, in_channels]
#filter: A Tensor. Must have the same type as input. Shape [filter_depth, filter_height, filter_width, in_channels, out_channels].
    with tf.variable_scope(name) as scope:
        if w_init=="xavier":
            weights = tf.get_variable("weights",shape,initializer=tf.contrib.layers.xavier_initializer())
        elif w_init=="constant":
            weights = tf.get_variable("weights",shape,initializer=tf.constant_initializer(0))
        elif w_init=="normal":
            weights = tf.get_variable("weights",initializer=tf.random_normal(shape,mean=mean,stddev=stddev))
        else:
            raise ValueError("The  initialization "+ w_init+ " is invalid for the weights of the layer "+name)

        if b_init=="constant":
            biases = tf.get_variable("biases", [shape[4]], initializer=tf.constant_initializer(b_value))
        elif b_init=="xavier":
            biases= tf.get_variable("biases", [shape[4]], initializer=tf.contrib.layers.xavier_initializer())
        else:
            raise ValueError("The  initialization "+ b_init+ " is invalid for the bias of the layer "+name)

        conv=tf.nn.conv3d(x, weights, [1, stride[0], stride[1],stride[2], 1],padding=padding)
        conv=tf.nn.bias_add(conv, biases)
        if batch_norm:
            conv=tf.contrib.layers.batch_norm(conv,activation_fn=None,is_training=training,scope=scope,reuse=False)

        if activation:
            activ =tf.nn.relu(conv)
        else:
            activ=conv
    return activ

def conv_layer_load(name,x,new_weights,new_biases,stride=1, padding='SAME',trainable = True,activation=True,batch_norm=False,training=False):
    with tf.variable_scope(name) as scope:
        weights=tf.get_variable("weights",initializer=tf.constant(new_weights),trainable = trainable)
        biases=tf.get_variable("biases",initializer=tf.constant(new_biases),trainable = trainable)
        conv=tf.nn.conv2d(x, weights, [1, stride, stride, 1],padding=padding)
        conv=tf.nn.bias_add(conv, biases)
        if batch_norm:
            #conv=CustomBatchNormalization(conv, tf.constant(True), scope=scope, reuse=False,reuse_averages=True)
            conv=tf.contrib.layers.batch_norm(conv,activation_fn=None,is_training=training,scope=scope,reuse=None)
        if activation:
            activ = tf.nn.relu(conv)
        else:
            activ=conv
    return activ

def conv_layer_reuse(name,x,stride=[1,1], padding='SAME',activation=True,batch_norm=True,training=False):
    with tf.variable_scope(name, reuse=True) as scope:
        weights=tf.get_variable("weights")
        biases=tf.get_variable("biases")
        conv=tf.nn.conv2d(x, weights, [1, stride[0], stride[1], 1],padding=padding)
        conv=tf.nn.bias_add(conv, biases)
        if batch_norm:
            #conv=CustomBatchNormalization(conv,tf.constant(False),scope=scope, reuse=True,reuse_averages=True)
            conv=tf.contrib.layers.batch_norm(conv,activation_fn=None,is_training=training,scope=scope,reuse=True)

        if activation:
            activ = tf.nn.relu(conv)
        else:
            activ=conv
    return activ

def deconv_layer_reuse(name,x,output_channel,filter_size=3,stride=2,output_shape=False, padding='SAME',):
    batch_size=x.get_shape()[0]
    hSize=x.get_shape()[1]
    wSize=x.get_shape()[2]
    nclasses=x.get_shape()[3]
    with tf.variable_scope(name, reuse=True) as scope:
        deconv_filter=tf.get_variable("up_filter")
        batch_size=x.get_shape()[0]
        nclasses=x.get_shape()[3]
        h = ((hSize - 1) * stride) + filter_size
        w = ((wSize - 1) * stride) + filter_size
        if not output_shape:
            output_shape=(int(batch_size),int(h),int(w),int(output_channel))
        deconv = tf.nn.conv2d_transpose(x, deconv_filter,output_shape ,strides=[1,stride,stride, 1], padding=padding)
    return deconv

def deconv_layer(name,x,output_channel,filter_size=3,stride=2,output_shape=False, padding='SAME',trainable=True,w_init="xavier"):
    batch_size=x.get_shape()[0]
    hSize=x.get_shape()[1]
    wSize=x.get_shape()[2]
    nclasses=x.get_shape()[3]
    width =filter_size
    heigh = filter_size
    if w_init=="bilinear":
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([filter_size, filter_size])
        for w in range(width):
            for h in range(heigh):
                value = (1 - abs(w / f - c)) * (1 - abs(h / f - c))
                bilinear[w, h] = value
        weights = np.zeros([filter_size,filter_size,nclasses,nclasses],dtype=np.float32)
        for i in range(nclasses):
            weights[:, :, i, i] = bilinear
        with tf.variable_scope(name) as scope:
            deconv_filter= tf.get_variable("up_filter", initializer=tf.constant(weights),trainable=trainable)
    elif (w_init=="xavier"):
        with tf.variable_scope(name) as scope:
            deconv_filter= tf.get_variable("up_filter", [filter_size,filter_size,output_channel,nclasses],initializer=tf.contrib.layers.xavier_initializer_conv2d())
            deconv_bias=tf.get_variable("biases", [output_channel], initializer=tf.constant_initializer(0.0))

    h = ((hSize - 1) * stride) + filter_size
    w = ((wSize - 1) * stride) + filter_size
    if  not output_shape:
        output_shape=(int(batch_size),int(h),int(w),int(output_channel))

    deconv = tf.nn.conv2d_transpose(x, deconv_filter,output_shape ,strides=[1,stride,stride, 1], padding=padding)
    deconv=tf.nn.relu(deconv+deconv_bias)
        #print(deconv.get_shape(),output_shape,x.get_shape())

    return deconv


def deconv3d_layer(name,x,output_channel,filter_size=64,stride=[1,1,1],output_shape=False, padding='VALID',trainable=True,w_init="xavier",activation=True):
    batch_size=x.get_shape()[0]
    dSize=x.get_shape()[1]
    hSize=x.get_shape()[2]
    wSize=x.get_shape()[3]
    nclasses=x.get_shape()[4]
    width =filter_size
    heigh = filter_size

    with tf.variable_scope(name) as scope:
        deconv_filter= tf.get_variable("up_filter", [filter_size,filter_size,filter_size,output_channel,nclasses],initializer=tf.contrib.layers.xavier_initializer_conv2d())
        deconv_bias=tf.get_variable("biases", [output_channel], initializer=tf.constant_initializer(0.0))

    h = ((hSize - 1) * stride[1]) + filter_size
    w = ((wSize - 1) * stride[2]) + filter_size
    d= ((dSize - 1) * stride[0]) + filter_size
    if  not output_shape:
        output_shape=(int(batch_size),int(d),int(h),int(w),int(output_channel))

    deconv = tf.nn.conv3d_transpose(x, deconv_filter,output_shape ,strides=[1,stride[0],stride[1],stride[2], 1], padding=padding)
    deconv=deconv+deconv_bias
    if activation:
        deconv=tf.nn.relu(deconv)

    #print(deconv.get_shape(),output_shape,x.get_shape())

    return deconv

def pool_layer(name,x,pool_type="max",ksize=2,stride_w=2,strid_h=2,padding='SAME'):
    with tf.variable_scope(name) as scope:
        if pool_type=="max":
            pool=tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride_w, strid_h, 1],
                              padding=padding)
        elif pool_type=="avg":
            pool=tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride_w, strid_h, 1],
                              padding=padding)
        else:
            raise ValueError("The  type "+ pool_type+ " is invalid for the layer "+name)
    return pool


def correlation_map(x, y, max_disp):
    corr_tensors = []
    for i in range(0,max_disp):
        temp=y[::,::,0:int(y.get_shape()[2])-i,::]
        shifted = tf.pad(temp,[[0, 0], [0, 0], [i, 0], [0, 0]], "CONSTANT")
        corr = tf.reduce_sum(tf.multiply(shifted, x), axis=3)

        corr_tensors.append(corr)
    # for i in range(max_disp + 1):
    #     shifted = tf.pad(tf.slice(x, [0, 0, i, 0], [-1]*4),
    #                      [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
    #     corr = tf.reduce_mean(tf.multiply(shifted, y), axis=3)
    #     corr_tensors.append(corr)
    return tf.transpose(tf.stack(corr_tensors),
                        perm=[1, 2, 3, 0])

def smoothL1(y_true, y_pred,HUBER_DELTA=1):
    x = tf.abs(y_true - y_pred)
    x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  x

def loss(name,x,y,loss_type="softmax",HUBER_DELTA=1):
    #epsilon = tf.constant(value=1e-5)

    #print(reshaped_x.get_shape(),reshaped_y.get_shape())
    #exit(0)
    with tf.variable_scope(name) as scope:
        if loss_type=="softmax":
            nclasses=int(x.get_shape()[3])
            reshaped_x =tf.reshape(x, (-1, nclasses))
            reshaped_y = tf.reshape(y, (-1, nclasses))

            loss=tf.nn.softmax_cross_entropy_with_logits(logits=reshaped_x, labels=reshaped_y)
        elif loss_type=="cross_entropy":
            sum_log=tf.reduce_sum(reshaped_x,[1])
            logits = reshaped_x/sum_log
            #softmax=tf.nn.softmax(reshaped_x) + epsilon
            loss=-tf.reduce_sum(reshaped_y * tf.log(logits), reduction_indices=[1])
        elif loss_type=="softmax_sparse":
            nclasses=int(x.get_shape()[3])

            valid_labels=tf.not_equal(y, 0)
            x=tf.boolean_mask(x, valid_labels)
            y=tf.boolean_mask(y, valid_labels)
            reshaped_x =tf.reshape(x, [-1,nclasses])
            reshaped_y = tf.reshape(y, [-1])
            loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_x, labels=reshaped_y, name=None)
            cost=tf.reduce_mean(loss)
        elif loss_type=="regression":
            #abs_dif=tf.pow(x-y, 2)
            #valid_labels=tf.not_equal(y, 0,name="not_equal")
            #nsamples=tf.reduce_sum(tf.cast(valid_labels, tf.float32),name="nsamples")
            #reshaped_x=tf.boolean_mask(x, valid_labels,name="reshaped_x")
            #reshaped_y=tf.boolean_mask(y, valid_labels,name="reshaped_y")
            #print("loss",x.get_shape(),y.get_shape(),valid_labels.get_shape(),reshaped_x.get_shape(),reshaped_y.get_shape())
            #exit(0)
            #n_samples=tf.size(x)
            # Mean squared error
            #loss = tf.reduce_sum(sparse_abs_dif)#/(2*n_samples)

            x=tf.squeeze(x,axis=3)
            mask = tf.cast(y>0, dtype=tf.bool)
            loss_ = tf.abs(tf.subtract(x, y))
            #loss_=smoothL1(x, y,HUBER_DELTA=HUBER_DELTA)
            loss_ = tf.where(mask, loss_, tf.zeros_like(loss_))
            #x=tf.where(y>0,tf.squeeze(x,axis=3),y)
            #x=tf.where(y>0,x,y)
            #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()  if 'bias' not in v.name ]) * 0.001
            cost=tf.reduce_mean(loss_)#+ lossL2 #/int(x.get_shape()[1]*x.get_shape()[2])

        else:
            raise ValueError("The  loss type "+ loss_type+ " is invalid for the loss layer "+name)
        #cost = tf.reduce_mean(loss)


    #logits function does not normalize the costs to have sum=1
    return cost

def optimizer(loss,learning_rate,optimizer_type="GradientDescent",beta1=0.9, beta2=0.99, epsilon=1e-08,rho=0.95, initial_accumulator_value=0.1,
momentum=0.9):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if optimizer_type=="GradientDescent":
            optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        elif optimizer_type=="Adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1, beta2=beta2).minimize(loss)
        elif optimizer_type=="AdaDelta":
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,rho=rho, epsilon=epsilon).minimize(loss)
        elif optimizer_type=="AdaGrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate,initial_accumulator_value=initial_accumulator_value).minimize(loss)
        elif optimizer_type=="Momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum).minimize(loss)
        else:
            raise ValueError("The  optimizer type "+ optimizer_type+ " is invalid")

        return optimizer

def dot_product(deconv_left, deconv_right,nclasses):
    res=[]
    shape=[int(shape) for shape in deconv_left.get_shape()]
    r_width= int(deconv_right.get_shape()[2])

    for loc_idx in range(nclasses):
        #left_features= deconv_left[:,:,max(start_id,-x_off):min(end_id,left_cols),:]
        right_features=deconv_right[:,:,r_width-loc_idx-shape[2]:r_width-loc_idx,:]

        multiplication=tf.multiply(deconv_left,right_features)
        inner_product=tf.reduce_sum(multiplication,-1)
        res.append(inner_product)

    res_tensor=tf.stack(res,name='concat')
    res_tensor=tf.transpose(res_tensor,perm=[1,2,3,0])
    return res_tensor

    # for n in range(shape[0]):
    #     left_features= tf.expand_dims(deconv_left[n,:,:,:],-1)
    #     right_features=tf.expand_dims(deconv_right[n,:,:,:],0)
    #     inner_product=tf.nn.conv2d(right_features,left_features,  [1, 1, 1, 1],padding="VALID")
    #     res.append(inner_product)
    #
    # res_tensor=tf.stack(res,name='concat')
    # return tf.squeeze(res_tensor,[4])
    # dist=[]
    #
    # for n in range(nclasses):
    #     dist.append(tf.reduce_sum(-tf.abs(tf.subtract(deconv_left,deconv_right[:,:,n:n+shape[2],:])), axis=[2,3]))
    # dist_tensor=tf.stack(dist)
    # res=tf.transpose(dist_tensor,perm=[1,2,0])
    # res=tf.expand_dims(res,1)
    #
    # return res


    #subtr=-tf.abs(tf.subtract(deconv_left,deconv_right))
    #inner_product=tf.expand_dims(tf.reduce_sum(subtr, axis=3),1)
    #return inner_product

def correlation_layer(name,deconv_left, deconv_right,nclasses,w_init="xavier",b_init="xavier"):
    batch=int(deconv_left.get_shape()[0])
    receptive_field_w=int(deconv_left.get_shape()[1])
    receptive_field_h=int(deconv_left.get_shape()[2])
    feature_size=int(deconv_right.get_shape()[3])*2
    r_width= int(deconv_right.get_shape()[2])
    res=[]
    for disp in range(nclasses):
        res.append(tf.concat([deconv_left,deconv_right[:,:,r_width-disp-receptive_field_h:r_width-disp,:]],3))

    features_tensor=tf.stack(res)
    features_tensor=tf.transpose(features_tensor,[1,2,3,0,4])
    features_tensor=tf.reshape(features_tensor,[-1,1,nclasses,feature_size])

    shape=[1,3,feature_size,feature_size]
    shape2=[1,3,feature_size,1]
    with tf.variable_scope(name) as scope:
        if w_init=="xavier":
                weights = tf.get_variable("weights",shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
                weights2 = tf.get_variable("weights2",shape2,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        elif w_init=="constant":
            weights = tf.get_variable("weights",shape,initializer=tf.constant_initializer(0))
            weights2 = tf.get_variable("weights2",shape2,initializer=tf.constant_initializer(0))

        elif w_init=="normal":
            weights = tf.get_variable("weights",initializer=tf.random_normal(shape,mean=mean,stddev=stddev))
            weights2 = tf.get_variable("weights2",shape2,initializer=tf.contrib.layers.xavier_initializer_conv2d())

        else:
            raise ValueError("The  initialization "+ w_init+ " is invalid for the weights of the layer "+name)

        if b_init=="constant":
            biases = tf.get_variable("biases", [shape[3]], initializer=tf.constant_initializer(b_value))
        elif b_init=="xavier":
            biases= tf.get_variable("biases", [shape[3]], initializer=tf.contrib.layers.xavier_initializer())
        else:
            raise ValueError("The  initialization "+ b_init+ " is invalid for the bias of the layer "+name)

        conv=tf.nn.conv2d(features_tensor, weights, [1, 1, 1, 1],padding="SAME")
        conv=tf.nn.bias_add(conv, biases)
        activ =tf.nn.relu(conv)

        conv=tf.nn.conv2d(activ, weights2, [1, 1, 1, 1],padding="SAME")

    conv=tf.reshape(conv,[batch,receptive_field_w,receptive_field_h,nclasses])

    return conv

def correlation_layer_test(name,deconv_left, deconv_right,nclasses,w_init="xavier",b_init="xavier"):
    #]initialize parameters
    batch=int(deconv_left.get_shape()[0])
    receptive_field=int(deconv_left.get_shape()[1])
    feature_size=int(deconv_right.get_shape()[3])*2
    r_width= int(deconv_right.get_shape()[2])
    shape=[1,3,feature_size,feature_size]
    shape2=[1,3,feature_size,1]
    with tf.variable_scope(name) as scope:
        if w_init=="xavier":
                weights = tf.get_variable("weights",shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        elif w_init=="constant":
            weights = tf.get_variable("weights",shape,initializer=tf.constant_initializer(0))
        elif w_init=="normal":
            weights = tf.get_variable("weights",initializer=tf.random_normal(shape,mean=mean,stddev=stddev))
            weights2 = tf.get_variable("weights2",shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())

        else:
            raise ValueError("The  initialization "+ w_init+ " is invalid for the weights of the layer "+name)

        if b_init=="constant":
            biases = tf.get_variable("biases", [shape[3]], initializer=tf.constant_initializer(b_value))
        elif b_init=="xavier":
            biases= tf.get_variable("biases", [shape[3]], initializer=tf.contrib.layers.xavier_initializer())
        else:
            raise ValueError("The  initialization "+ b_init+ " is invalid for the bias of the layer "+name)


    res=[]
    for disp in range(nclasses):
        left_features= deconv_left[:,:,disp:r_width,:]
        right_features=deconv_right[:,:,0:r_width-disp,:]
        concat_feat=tf.concat([left_features,right_features],3)+10e5
        concat_feat=tf.pad(concat_feat, [[0,0], [0,0],[disp,0],[0,0]], "CONSTANT")-10e5
        res.append(concat_feat)

    features_tensor=tf.stack(res)
    features_tensor=tf.transpose(features_tensor,[1,2,3,0,4])
    features_tensor=tf.reshape(features_tensor,[-1,1,nclasses,feature_size])

    shape=[1,3,feature_size,feature_size]
    with tf.variable_scope(name) as scope:
        if w_init=="xavier":
                weights = tf.get_variable("weights",shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        elif w_init=="constant":
            weights = tf.get_variable("weights",shape,initializer=tf.constant_initializer(0))
        elif w_init=="normal":
            weights = tf.get_variable("weights",initializer=tf.random_normal(shape,mean=mean,stddev=stddev))
        else:
            raise ValueError("The  initialization "+ w_init+ " is invalid for the weights of the layer "+name)

        if b_init=="constant":
            biases = tf.get_variable("biases", [shape[3]], initializer=tf.constant_initializer(b_value))
        elif b_init=="xavier":
            biases= tf.get_variable("biases", [shape[3]], initializer=tf.contrib.layers.xavier_initializer())
        else:
            raise ValueError("The  initialization "+ b_init+ " is invalid for the bias of the layer "+name)

        conv=tf.nn.conv2d(features_tensor, weights, [1, 1, 1, 1],padding="SAME")
        conv=tf.nn.bias_add(conv, biases)
        activ =tf.nn.relu(conv)

        shape=[1,3,feature_size,1]
        weights2 = tf.get_variable("weights2",shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv=tf.nn.conv2d(activ, weights2, [1, 1, 1, 1],padding="SAME")

    conv=tf.reshape(conv,[batch,receptive_field,r_width,nclasses])

    return conv

def dot_product_test(deconv_left, deconv_right,batch_size,image_rows,image_col, receptive_field,nclasses):
    half_rf=int(receptive_field/2)
    #shape=[int(shape) for shape in deconv_left.get_shape()]
    left_cols=image_col

    #x_left=tf.pad(deconv_left, [[0,0],[half_rf,half_rf], [half_rf,half_rf],[0,0]], "CONSTANT")
    #x_right=tf.pad(deconv_right, [[0,0],[half_rf,half_rf], [half_rf+maxDisp,half_rf],[0,0]], "CONSTANT")
    res=np.ones((batch_size,image_rows,image_col,nclasses))*(-1e9)
    #res = tf.Variable(tf.zeros([batch_size,image_rows,image_col,nclasses]), name="res")
    unary_vol=[]#np.zeros((int(batch),int(image_rows),int(left_cols),int(nclasses)))
    start_id=0
    simult_amount=image_col
    end_id = start_id+simult_amount#int(shape[2])

    while start_id<left_cols-1:
    #    print("start2",start_id)
        #unary_vol=[]
        for loc_idx in range(nclasses):

            x_off = -loc_idx # always <= 0
            #print("loca_ixs",loc_idx,start_id,end_id,x_off)
            if (end_id+x_off > 0):
                if (left_cols > start_id+x_off):
                    #print("range left",max(start_id,-x_off),min(end_id,shape[2]))
                    #print("range right",max(0,start_id+x_off),min(end_id+x_off,shape[2]+x_off))
                    left_features= deconv_left[:,:,max(start_id,-x_off):min(end_id,left_cols),:]
                    right_features=deconv_right[:,:,max(0,start_id+x_off):min(end_id+x_off,left_cols+x_off),:]
                    #print(left_features.get_shape(),right_features.get_shape())
                    multiplication=np.multiply(left_features,right_features)
                    inner_product=np.sum(multiplication,axis=3)#tf.reduce_sum(multiplication,-1)
                    #subtr=-np.abs(np.subtract(left_features,right_features))
                    #inner_product=np.sum(subtr, axis=3)
                    #print(inner_product.get_shape())
                    #print("left",loc_idx,max(start_id,-x_off),min(end_id,left_cols))
                    #print("right",loc_idx,max(0,start_id+x_off),min(end_id+x_off,left_cols+x_off))
                    #print("inner_product", inner_product.shape)
                    #print(inner_product.shape,max(start_id,-x_off),min(end_id,left_cols))

                    res[:,:,max(start_id,-x_off):min(end_id,left_cols),loc_idx]=inner_product

                    #print(inner_product.get_shape())
                    #if(start_id+x_off<0):
                    #    pad=loc_idx#simult_amount-int(inner_product.get_shape()[2])
                    #    inner_product=tf.pad(inner_product, [[0,0], [0,0],[pad,0]], "CONSTANT")
                    #unary_vol.append(inner_product)
                    #print(inner_product.get_shape())
                    # #inner_product=tf.nn.conv2d(right_features,left_features,  [1, 1, 1, 1],padding="VALID")
                    # #unary_vol[:,:,max(start_id,-x_off+1):min(end_id,left_cols-x_off),loc_idx]=inner_product

                    #unary_vol.append(inner_product)
                    #print(start_id,loc_idx,inner_product.get_shape())
                    #unary_vol.append(inner_product)
                    # counter+=1
                    #start_id = end_id + 1
                    #end_id = min(left_cols, end_id+maxDisp)
            #else:
                #unary_vol.append(np.zeros((1,image_rows,simult_amount)))

        #print(len(unary_vol),unary_vol[0].get_shape())
        #if start_id==0:
        #    total_volume=tf.stack(unary_vol)
        #else:
        #    total_volume=tf.concat([total_volume,tf.stack(unary_vol)],3)
        start_id=end_id
        end_id+=simult_amounttf.image.resize_images
        #print(total_volume.get_shape())

    #for i in range(len(unary_vol)):#pad the cols on the left for border cases

    #reshaped= tf.transpose(total_volume, perm=[1,2,3,0])
    #print(reshaped.get_shape())
    #a=tf.stack(unary_vol)
    #print("final",a.get_shape())
    return res

if __name__ == '__main__':
    import Image, ImageDraw
    import matplotlib.pyplot as plt
    import random


    im1 = (scp.imread("./alexnet/poodle.png")[:,:,:3]).astype(np.float32)
    num_classes=10
    im_gt=np.zeros((114,114,num_classes))

    x = tf.placeholder(tf.float32, (None,)+ im1.shape)
    y = tf.placeholder(tf.float32,  (None,)+ im_gt.shape)
    #plt.imshow(vis_square(data_dict["conv1_1"][0]))
    #plt.show()
    conv1=conv_layer_load("conv1",x,data_dict["conv1_1"][0],data_dict["conv1_1"][1]) #conv_layer("conv1",x,[11,11,3,96])
    pool=pool_layer("pool1",conv1)
    print("AHHAHAH",pool.get_shape())
    conv2=conv_layer_init("conv2",pool,[3,3,64,num_classes])
    loss=loss("loss", conv2,y)
    optimizer =optimizer(loss,0.1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        #with tf.variable_scope("conv1", reuse=True) as scope:
        #    print("teste",tf.get_variable("weights").eval())
        im1 = im1[np.newaxis,...]
        im_gt = im_gt[np.newaxis,...]
        imageSize=227
        maxNoise=20
        for i in range(10):
            # image = Image.new('RGB', (imageSize, imageSize))
            # im_gt=np.zeros([imageSize,imageSize],np.int32)
            # polypSizeX=random.randrange(15, 140)
            # polypSizeY=random.randrange(15, 140)
            # firstX=random.randrange(0,imageSize-polypSizeX)
            # firstY=random.randrange(0,imageSize-polypSizeY)
            # if(i % 2 == 0):
            #     draw = ImageDraw.Draw(image)
            #     draw.ellipse((polypSizeY, firstY, polypSizeY+polypSizeX, firstY+polypSizeY), fill = 'brown')
            # temp=np.asarray(image)
            # im_gt[temp[:,:,0]>0]=255
            # noise=np.random.rand(imageSize,imageSize,3)*maxNoise-(maxNoise/2.0)
            # noise=noise.astype(np.uint)
            # image+=noise
            sess.run(optimizer, feed_dict={x: im1, y: im_gt})
        writer = tf.train.SummaryWriter("./", graph=tf.get_default_graph())
