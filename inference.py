import train_kitty as kt
import argparse
import imageio
import numpy as np
import tensorflow as tf
from skimage.transform import resize


parser = argparse.ArgumentParser(
description='perform inference for hapnet')
parser.add_argument('left',help='path to left image.')
parser.add_argument('right',help='path to right image.')
parser.add_argument('model',help='path to pretrained model file.')
parser.add_argument('out',help='path to store the resulting disparity.')




def load_sample(left_path, right_path, size):

    batch_left=np.zeros((0, size[0],size[1],3), dtype=np.float32)
    batch_right=np.zeros((0, size[0],size[1],3), dtype=np.float32)



    #load images
    left = imageio.imread(left_path)
    right = imageio.imread(right_path)



    # resize images
    left = resize(left, size)
    right = resize(right, size)

    #normalize images
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    left=np.subtract(left,mean)/std
    right=np.subtract(right,mean)/std


    #create a batch of size 1
    left = left[np.newaxis,...]
    right = right[np.newaxis,...]

    batch_left=np.concatenate([batch_left, left],axis=0)
    batch_right=np.concatenate([batch_right, right],axis=0)


    return batch_left,batch_right

if __name__ == '__main__':    
    args = parser.parse_args()

    rescale = original_size[1]/size[1]


    batch=1
    scale=1.5
    maxDisp=128
    nclasses=maxDisp+1
    
    original_size = imageio.imread(args.left).shape[:2]
    size = [int(original_size[0]//scale), int(original_size[1]//scale)]
    

    x_left = tf.compat.v1.placeholder(tf.float32, (batch,size[0]//scale, size[1]//scale,3))
    x_right = tf.compat.v1.placeholder(tf.float32, (batch,size[0]//scale, size[1]//scale,3))
    y = tf.compat.v1.placeholder(tf.float32,  (batch,size[0]//scale,size[1]//scale))

    scoring,final_score_1,final_score_2,final_score_3=kt.init_shallow(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=False)

    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    saver = tf.train.Saver()

    with tf.Session(config = config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        saver.restore(sess, args.model)

        left, right = load_sample(args.left, args.right, size)

        pred,pred1,pred2,pred3=sess.run([scoring,final_score_1,final_score_2,final_score_3], feed_dict={x_left: left,x_right:right})
        
        pred=np.squeeze(pred)
        maxv=255
        minv=0
        pred = resize(pred, original_size)
        pred = pred * scale
        pred[pred>maxv]=maxv
        pred[pred<minv]=minv
        
        imageio.imsave(args.out, pred.astype(np.uint8))