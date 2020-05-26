import pdb
import sys
sys.path.insert(0, '../vgg')

sys.path.insert(0, '../')
import tensorflow as tf
import numpy as np
import cnn_function as cnn
import os
import random
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
#from tensorflow.python.client import device_lib
#print (device_lib.list_local_devices())
#exit(0)
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy
from util import init_logger, ft3d_filenames

def load_kitti_2012(kitti_2012_path,nvalidation=32,shuffle=True):
    #validation_index=np.random.randint(low=0, high=193, size=nvalidation)
    deck = list(range(0, 194))
    np.random.shuffle(deck)
    validation_index=deck[0:nvalidation]
    training_list_left=[]
    training_list_right=[]
    training_list_noc_label=[]
    validation_list_left=[]
    validation_list_right=[]
    validation_list_noc_label=[]

    for i in range(0,194):
        if i in validation_index:
            validation_list_left.append(os.path.join(kitti_2012_path, "training/colored_0",str(i).zfill(6)+"_10.png"))
            validation_list_right.append(os.path.join(kitti_2012_path, "training/colored_1",str(i).zfill(6)+"_10.png"))
            validation_list_noc_label.append(os.path.join(kitti_2012_path, "training/disp_noc",str(i).zfill(6)+"_10.png"))
            #print(os.path.exists(training_list_noc_label[-1]),os.path.exists(training_list_noc_label[-1]))
            #test=np.round(ndimage.imread(training_list_noc_label[-1])/256.0)

            #exit(0)

        else:
            training_list_left.append(os.path.join(kitti_2012_path, "training/colored_0",str(i).zfill(6)+"_10.png"))
            training_list_right.append(os.path.join(kitti_2012_path, "training/colored_1",str(i).zfill(6)+"_10.png"))
            training_list_noc_label.append(os.path.join(kitti_2012_path, "training/disp_noc",str(i).zfill(6)+"_10.png"))
            #print(os.path.exists(validation_list_noc_label[-1]),os.path.exists(validation_list_noc_label[-1]))
            #test=np.round(ndimage.imread(validation_list_noc_label[-1])/256.0)


    #test_list_left=[]
    #test_list_right=[]
    #for i in range(0,195):
    #    test_list_left.append(os.path.join(kitti_2012_path, "testing/colored_0",str(i).zfill(6)+"_10.png"))
    #    test_list_right.append(os.path.join(kitti_2012_path, "testing/colored_1",str(i).zfill(6)+"_10.png"))
    #    print(os.path.exists(test_list_left[-1]),os.path.exists(test_list_right[-1]))
    return training_list_left,training_list_right,training_list_noc_label,validation_list_left,validation_list_right,validation_list_noc_label


def load_kitti_2015(kitti_2015_path,nvalidation=40,shuffle=True):
    #validation_index=np.random.randint(low=0, high=199, size=nvalidation)
    deck = list(range(0, 200))
    np.random.shuffle(deck)
    validation_index=deck[0:nvalidation]
    training_list_left=[]
    training_list_right=[]
    training_list_noc_label=[]
    validation_list_left=[]
    validation_list_right=[]
    validation_list_noc_label=[]
    for i in range(0,200):
        if i in validation_index:
            validation_list_left.append(os.path.join(kitti_2015_path, "training/image_2",str(i).zfill(6)+"_10.png"))
            validation_list_right.append(os.path.join(kitti_2015_path, "training/image_3",str(i).zfill(6)+"_10.png"))
            validation_list_noc_label.append(os.path.join(kitti_2015_path, "training/disp_noc_0",str(i).zfill(6)+"_10.png"))
            #print(os.path.exists(validation_list_left[-1]),os.path.exists(validation_list_right[-1]),os.path.exists(training_list_noc_label[-1]))
            #test=np.round(ndimage.imread(training_list_noc_label[-1])/256.0)
            #test=np.round(ndimage.imread(training_list_noc_label[-1])/256.0)
            #if(np.max(test)>200):
            #if(np.max(test)>150):
            #    print(np.max(test))
                #exit(0)

        else:
            training_list_left.append(os.path.join(kitti_2015_path, "training/image_2",str(i).zfill(6)+"_10.png"))
            training_list_right.append(os.path.join(kitti_2015_path, "training/image_3",str(i).zfill(6)+"_10.png"))
            training_list_noc_label.append(os.path.join(kitti_2015_path, "training/disp_noc_0",str(i).zfill(6)+"_10.png"))
            #print(os.path.exists(training_list_left[-1]),os.path.exists(training_list_right[-1]),os.path.exists(training_list_noc_label[-1]))
            #test=np.round(ndimage.imread(validation_list_noc_label[-1])/256.0)
            #if(np.max(test)>150):
            #print(np.min(test),np.max(test))

    return training_list_left,training_list_right,training_list_noc_label,validation_list_left,validation_list_right,validation_list_noc_label

def load_disp(path):
    return np.round(ndimage.imread(path,flatten=True)/256.0)

#outdated ignore
def pad_image_and_label(im_left,im_right,im_d,total_patch_size,patch_size,maxDisp):
    constValue=-134.0/79.0
    if (im_left.shape[0]<total_patch_size) or(im_left.shape[1]<total_patch_size):
        pad_x_left=max(0,total_patch_size-im_left.shape[0])
        pad_y_left=max(0,total_patch_size-im_left.shape[1])
        im_left=np.pad(im_left, ((0,pad_x_left), (0,pad_y_left),(0,0)), "constant",constant_values=(constValue))

        im_right=np.pad(im_right,((0,pad_x_left), (0,pad_y_left),(0,0)), "constant",constant_values=(constValue))

        pad_x_d=max(0,patch_size-im_d.shape[0])
        pad_y_d=max(0,patch_size-im_d.shape[1])
        im_d=np.pad(im_d, ((0,pad_x_d), (pad_y_d,0)), "constant")


    if(im_right.shape[0]<total_patch_size) or(im_right.shape[1]<(total_patch_size+maxDisp)):
        pad_x_right=max(0,total_patch_size-im_right.shape[0])
        pad_y_right=max(0,maxDisp+total_patch_size-im_right.shape[1])
        im_right=np.pad(im_right, ((0,pad_x_right),(pad_y_right,0),(0,0)), "constant",constant_values=(constValue))
    return im_left,im_right,im_d


def load_full_images(training_list_left,training_list_right,training_list_noc_label,maxDisp,batch_size,size):
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    batch_left=[]
    batch_right=[]
    batch_disp=[]

    #size=[384,1248]
    batch_left=np.zeros((0, size[0],size[1],3), dtype=np.float32)
    batch_right=np.zeros((0, size[0],size[1],3), dtype=np.float32)
    batch_disp=np.zeros((0, size[0],size[1]), dtype=np.float32)

    #negative_mining=[2, 3, 5, 6, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 69, 72, 73, 74, 88, 93, 100, 101, 102, 103, 104, 107, 108, 109, 110, 118, 120, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 142, 143, 144, 145, 147, 148, 150, 151, 152, 153, 154, 155, 158, 159, 160, 161, 162, 163, 164, 165, 169, 172, 173, 176, 178, 183, 187, 188, 192, 193, 194, 195, 197]
    for batch in range(0,batch_size):
        random_image_index=np.random.randint(len(training_list_left))
        #random_image_index=np.random.randint(len(negative_mining))
        #random_image_index=negative_mining[random_image_index]
        #d=disp_image_to_label(training_list_noc_label[random_image_index],maxDisp+1)

        orign_size=training_list_left[random_image_index].shape
        r_pad=size[0]-orign_size[0]
        c_pad=size[1]-orign_size[1]

        left=np.pad(training_list_left[random_image_index],[(r_pad, 0), (0,c_pad),(0, 0)], mode='constant')
        right=np.pad(training_list_right[random_image_index],[(r_pad, 0), (0,c_pad),(0, 0)],mode= 'constant')
        d=np.pad(training_list_noc_label[random_image_index],[(r_pad, 0), (0,c_pad)],mode= 'constant')

        #orign_size=training_list_left[random_image_index].shape
        #random_x=np.random.randint(orign_size[0]-size[0])
        #random_y=np.random.randint(orign_size[1]-size[1])
        #left=training_list_left[random_image_index][random_x:random_x+size[0],random_y:random_y+size[1],:]
        #right=training_list_right[random_image_index][random_x:random_x+size[0],random_y:random_y+size[1],:]
        #d=np.round(training_list_noc_label[random_image_index])[random_x:random_x+size[0],random_y:random_y+size[1]]


        left=left/255.0
        right=right/255.0

        left=(left-mean)/std
        right=(right-mean)/std
        #left=(left-np.mean(left))/np.std(left)
        #right=(right-np.mean(right))/np.std(right)

        #temp
        # left=training_list_left[random_image_index][20:20+size[0],200:200+size[1],:]
        # right=training_list_right[random_image_index][20:20+size[0],200:200+size[1],:]
        # d=training_list_noc_label[random_image_index][20:20+size[0],200:200+size[1]]


        left = left[np.newaxis,...]
        right = right[np.newaxis,...]
        d= d[np.newaxis,...]
        batch_left=np.concatenate([batch_left, left],axis=0)
        batch_right=np.concatenate([batch_right, right],axis=0)
        batch_disp=np.concatenate([batch_disp, d],axis=0)
    #batch_left=tf.image.resize_images(tf.stack(batch_left),size)
    #batch_right=tf.image.resize_images(tf.stack(batch_right),size)
    #sbatch_disp=tf.squeeze(tf.image.resize_images(tf.expand_dims(tf.stack(batch_disp),-1),size))
    return batch_left,batch_right,batch_disp



def preprocess(left_img, right_img, target, size):
    left_img = tf.image.convert_image_dtype(left_img, tf.float32)
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    height, width = size
    orig_width = tf.shape(left_img)[1]
    left_img = left_img - mean
    left_img=left_img/std
    right_img = tf.image.convert_image_dtype(right_img, tf.float32)
    right_img = right_img - mean
    right_img=right_img/std

    #scale=0.4+0.8*np.random.random_sample()
    #newSize=[int(height*scale),int(width*scale)]
    random_x=np.random.randint(540-size[0])
    random_y=np.random.randint(960-size[1])

    left_img=left_img[random_x:random_x+size[0],random_y:random_y+size[1],:]
    right_img=right_img[random_x:random_x+size[0],random_y:random_y+size[1],:]
    target=target[random_x:random_x+size[0],random_y:random_y+size[1]]

    #mask = tf.cast(target<192, dtype=tf.bool)
    #target = tf.where(mask, target, tf.zeros_like(target))
    #
    # left_img = tf.image.resize_bilinear(left_img[np.newaxis, :, :, :], [height, width])[0]
    # right_img = tf.image.resize_bilinear(right_img[np.newaxis,:, :, :], [height, width])[0]
    # target =tf.image.resize_nearest_neighbor(target[np.newaxis, :, :, np.newaxis], [height, width])[0]
    # target=tf.squeeze(target,-1)
    #target = target /scale
    left_img.set_shape([height, width, 3])
    right_img.set_shape([height, width, 3])
    target.set_shape([height, width])
    return left_img, right_img, target


def load_numpy(path):
    a=np.load(path.decode("utf-8") )
    return a

def load_image(path):
    return ndimage.imread(path.decode("utf-8") )

def read_sample(filename_queue):
    filenames = filename_queue.dequeue()
    left_fn, right_fn, disp_fn = filenames[0], filenames[1], filenames[2]
    #left_img = tf.image.decode_image(tf.read_file(left_fn))
    #right_img = tf.image.decode_image(tf.read_file(right_fn))
    #atest=np.load
    left_img = tf.py_func(lambda x: load_image(x), [left_fn], [tf.uint8])[0]
    right_img = tf.py_func(lambda x: load_image(x), [right_fn], [tf.uint8])[0]
    target = tf.py_func(lambda x: load_numpy(x), [disp_fn], [tf.float32])[0]
    return left_img, right_img, target


def input_pipeline(filenames, input_size, batch_size, num_epochs=None,shuffle=True):

    filename_queue = tf.train.input_producer(
        filenames, element_shape=[3], num_epochs=num_epochs, shuffle=shuffle)

    left_img, right_img, target = read_sample(filename_queue)
    #left_img, right_img, target=load_scene_flow_images(filenames)
    left_img, right_img, target = preprocess(left_img, right_img, target, input_size)
    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    left_img_batch, right_img_batch, target_batch = tf.train.shuffle_batch(
        [left_img, right_img, target], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return left_img_batch, right_img_batch, target_batch


def load_scene_flow_images(training_list,training_list_noc_label,batch_size,size,negative_mining=None):
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    batch_left=[]
    batch_right=[]
    batch_disp=[]

    #size=[int(size[0]/2),int(size[1]/2)]
    batch_left=np.zeros((0, size[0],size[1],3), dtype=np.float32)
    batch_right=np.zeros((0, size[0],size[1],3), dtype=np.float32)
    batch_disp=np.zeros((0, size[0],size[1]), dtype=np.float32)

    for batch in range(0,batch_size):
        if (negative_mining):
            random_negative=np.random.randint(len(negative_mining))
            random_image_index=negative_mining[random_negative]
        else:
            random_image_index=np.random.randint(len(training_list))
        #d=disp_image_to_label(training_list_noc_label[random_image_index],maxDisp+1)
        #print(training_list[random_image_index],training_list_noc_label[random_image_index])
        random_x=np.random.randint(540-size[0])
        random_y=np.random.randint(959-size[1])
        left=ndimage.imread(training_list[random_image_index])[random_x:random_x+size[0],random_y:random_y+size[1],:]
        right=ndimage.imread(training_list[random_image_index].replace("left","right"))[random_x:random_x+size[0],random_y:random_y+size[1],:]
        d=np.round(np.load(training_list_noc_label[random_image_index]))[random_x:random_x+size[0],random_y:random_y+size[1]]
        d[d>192]=0
        #d[:,0:50]=0
        #left=scipy.misc.imresize(left,size)
        #right=scipy.misc.imresize(right,size)
        #d=scipy.misc.imresize(d,size,interp='nearest')
        #d[d>191]=0
        #depth[depth>128]=0
        #print(np.max(np.max(left)),np.min(np.min(left)),np.max(np.max(d)),np.min(np.min(d)))

        left=np.array(left,dtype=np.float32)
        right=np.array(right,dtype=np.float32)
        r_pad=size[0]-left.shape[0]
        c_pad=size[1]-left.shape[1]
        #
        left=np.pad(left,[(r_pad, 0), (0,c_pad),(0, 0)], mode='constant')
        right=np.pad(right,[(r_pad, 0), (0,c_pad),(0, 0)],mode= 'constant')
        d=np.pad(d,[(r_pad, 0), (0,c_pad)],mode= 'constant')
        # for i in range(int(size[0]*size[1]/2)):
        #     random_x=np.random.randint(size[0])
        #     random_y=np.random.randint(size[1])
        #     d[random_x,random_y]=0
        left=left/255
        right=right/255
        left=np.subtract(left,mean)/std#left-np.mean(left))/np.std(left)
        right=np.subtract(right,mean)/std#(right-np.mean(right))/np.std(right)

        #max_val=np.max(np.max(left))
        #min_val=np.min(np.min(left))
        #left=(left-np.mean(left))/np.std(left)
        #right=(right-np.mean(right))/np.std(right)

        #print(np.max(np.max(left)),np.min(np.min(left)),np.max(np.max(d)),np.min(np.min(d)))
        #scipy.misc.imsave("left.png",left[280:370,300:370,::])
        #scipy.misc.imsave("right.png",right[280:370,300-int(d[325,325]):370-int(d[325,325]),::])

        #scipy.misc.imsave("d.png",d)
        # print(np.max(np.max(left)),np.min(np.min(left)),np.max(np.max(d)),np.min(np.min(d)))
        # scipy.misc.imsave("left.png",left)
        # scipy.misc.imsave("right.png",right)
        # scipy.misc.imsave("depth.png",d)
        # exit(0)
        #preprocessing
        #print(training_list[random_image_index])
        # plt.figure(1)
        # plt.imshow(left)
        # plt.colorbar()
        # plt.figure(2)
        # plt.imshow(right)
        # plt.colorbar()
        # plt.figure(3)
        # print(training_list_noc_label[random_image_index])
        # plt.imshow(d)
        # plt.colorbar()
        # plt.show()
        #exit(0)
        left = left[np.newaxis,...]
        right = right[np.newaxis,...]
        d= d[np.newaxis,...]
        batch_left=np.concatenate([batch_left, left],axis=0)
        batch_right=np.concatenate([batch_right, right],axis=0)
        batch_disp=np.concatenate([batch_disp, d],axis=0)

    #sbatch_disp=tf.squeeze(tf.image.resize_images(tf.expand_dims(tf.stack(batch_disp),-1),size))

    return batch_left,batch_right,batch_disp


def load_random_patch(training_list_left,training_list_right,training_list_noc_label,receptive_field,maxDisp,batch_size,
valid_pixels_list):
    #total_patch_size=receptive_field
    half_path_size=int(receptive_field/2)
    #half_path_size_w=int(maxDisp+receptive_field/2)
    #half_receptive=int(patch_size/2)
    #half_max_disp=int(maxDisp/2)


    batch_left=np.zeros((0, receptive_field,receptive_field,3), dtype=np.float32)
    batch_right=np.zeros((0, receptive_field,maxDisp+receptive_field,3), dtype=np.float32)
    batch_disp=np.zeros((0, receptive_field,receptive_field), dtype=np.float32)

    for batch in range(0,batch_size):
        random_image_index=np.random.randint(len(training_list_left))

        #valid_choices=np.where(training_list_noc_label[random_image_index]!=0)
        #
        # #put this in a funciton or something
        # #transpose
        # valid_choices=list(map(list, zip(*valid_choices)))
        #
        # valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[0]>half_path_size]
        # valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[0]<training_list_left[random_image_index].shape[0]-half_path_size]
        #
        # valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[1]>half_path_size+half_max_disp+maxDisp]
        # valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[1]<training_list_left[random_image_index].shape[1]-half_max_disp-half_path_size]

        random_choice=np.random.randint(len(valid_pixels_list[random_image_index]))
        x_rand,y_rand=valid_pixels_list[random_image_index][random_choice][0],valid_pixels_list[random_image_index][random_choice][1]
        #left=training_list_left[random_image_index][max(0,x_rand-half_path_size):x_rand+half_path_size+1,max(0,y_rand-half_path_size):y_rand+half_path_size+1,:]
        left=training_list_left[random_image_index][x_rand-half_path_size:x_rand+half_path_size,
        y_rand-half_path_size:y_rand+half_path_size,:]
        d=training_list_noc_label[random_image_index][x_rand-half_path_size:x_rand+half_path_size,y_rand-half_path_size:y_rand+half_path_size]
        right=training_list_right[random_image_index][x_rand-half_path_size:x_rand+half_path_size,y_rand-half_path_size-maxDisp:y_rand+half_path_size,:]
        # plt.figure(0)
        # plt.imshow(left)
        # plt.figure(1)
        # plt.imshow(right)
        # plt.figure(2)
        # plt.imshow(d,cmap="gray")
        # print(d)
        # plt.show()
        #left=training_list_left[random_image_index][x_rand-half_path_size:x_rand+half_path_size,
        #y_rand-half_path_size_w:y_rand+half_path_size_w,:]
        #d=int(training_list_noc_label[random_image_index][x_rand,y_rand])
        #right=training_list_right[random_image_index][x_rand-half_path_size:x_rand+half_path_size,
        #y_rand-d-half_path_size_w-half_max_disp:y_rand-d+half_path_size_w+half_max_disp,:]
        #d=np.full((1,1),half_max_disp)
        #print("displacment",d)
        #left,right,d=pad_image_and_label(left,right,d,total_patch_size,patch_size,maxDisp)
        #scipy.misc.imsave("left.png",left)
        #scipy.misc.imsave("left_complete.png",training_list_left[random_image_index][x_rand-half_path_size:x_rand+half_path_size+1,:,:])
        #scipy.misc.imsave("right.png",right)
        #scipy.misc.imsave("right_complete.png",training_list_right[random_image_index][x_rand-half_path_size:x_rand+half_path_size+1,:,:])
        #exit(0)
        d=disp_image_to_label(d,maxDisp+1)
        # d=np.argmax(d,2)
        # scipy.misc.imsave("right_right.png",right[:,d[0]:d[0]+receptive_field,:])
        # exit(0)
        left = left[np.newaxis,...]
        right = right[np.newaxis,...]
        d= d[np.newaxis,...]

        #print(d.shape,batch_disp.shape)
        batch_left=np.concatenate([batch_left, left],axis=0)
        batch_right=np.concatenate([batch_right, right],axis=0)
        batch_disp=np.concatenate([batch_disp, d],axis=0)

    return batch_left,batch_right,batch_disp

def disp_image_to_label(disp_image,nclasses):
    #im_gt=np.zeros((disp_image.shape[0],disp_image.shape[1],nclasses),dtype=np.float32)
    #disp_image=nclasses-disp_image
    #for n in range(1,nclasses):
    #    im_gt[:,:,n]=(disp_image[:,:]==n)#*0.5
    disp_image[disp_image>nclasses-1]=0
    #disp_image[disp_image[:,:]==0]=-1
    #print(im_gt[:,:,1],np.max(im_gt[:,:,:]))

    # right_label=np.where(im_gt)
    # for n in range(0,len(right_label[0])):
    #    if right_label[2][n]>0:
    #        im_gt[right_label[0][n],right_label[1][n],right_label[2][n]-1]=0.2
    #
    #    if right_label[2][n]>2:
    #        im_gt[right_label[0][n],right_label[1][n],right_label[2][n]-2]=0.05
    #
    #    if right_label[2][n]<nclasses-1:
    #        im_gt[right_label[0][n],right_label[1][n],right_label[2][n]+1]=0.2
    #
    #    if right_label[2][n]<nclasses-2:
    #        im_gt[right_label[0][n],right_label[1][n],right_label[2][n]+2]=0.05

    #return im_gt
    return disp_image

def get_valid_pixels(training_list_noc_label,total_patch_size,maxDisp):
    half_path_size=int(total_patch_size/2)
    half_max_disp=int(maxDisp/2)
    list_of_valid_pixels=[]
    for i in range(len(training_list_noc_label)):
        valid_choices=np.where(training_list_noc_label[i]!=0)

        #put this in a funciton or something
        #transpose
        valid_choices=list(map(list, zip(*valid_choices)))

        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[0]>half_path_size]
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[0]<training_list_noc_label[i].shape[0]-half_path_size]
        #
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[1]>half_path_size+maxDisp+training_list_noc_label[i][x_y_pair[0]][x_y_pair[1]]]
        valid_choices=[x_y_pair for x_y_pair in valid_choices if x_y_pair[1]<training_list_noc_label[i].shape[1]-half_max_disp-half_path_size]
        list_of_valid_pixels.append(valid_choices)
    return list_of_valid_pixels

def load_all_images(left_paths,right_paths,disp_paths):
    left_images=[]
    right_images=[]
    disp_images=[]
    print("Loading and preprocessing images...")
    for i in range(len(left_paths)):
        left_images.append(processs_input_image(ndimage.imread(left_paths[i])))
        right_images.append(processs_input_image(ndimage.imread(right_paths[i])))
        disp_images.append(load_disp(disp_paths[i]))
    print("done!")
    return left_images,right_images,disp_images

def load_all_images_test(left_paths,right_paths):
    left_images=[]
    right_images=[]
    print("Loading and preprocessing images...")
    for i in range(len(left_paths)):
        left_images.append(processs_input_image(ndimage.imread(left_paths[i])))
        right_images.append(processs_input_image(ndimage.imread(right_paths[i])))
    print("done!")
    return left_images,right_images,

def processs_input_image(image):
    image=np.array(image,dtype=np.float32)
    #image[:,:,0]-=123.68
    #image[:,:,1]-=116.779
    #image[:,:,2]-=103.939
    #image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]
    #image=(image-np.mean(image))/np.std(image)
    return image#/79


def error(D_est,D_gt,error_margin=3,invalid_pixels=0):
    E = np.abs(D_gt-D_est);
    valid_pixels=E[D_gt!=invalid_pixels]
    temp=np.array(valid_pixels>error_margin,dtype=np.bool)
    #print(temp)
    n_err   = np.sum(temp);
    n_total = len(valid_pixels);
    return float(n_err)/float(n_total+1e-10)

def init_shallow_correlation(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])


    shape=[5,5,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)

    # shape=[3,3,64,128]
    # conv5_left=cnn.conv_layer_init("conv5",conv4_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    # shape=[3,3,128,128]
    # conv6_left=cnn.conv_layer_init("conv6",conv5_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    # conv6_1_left=cnn.conv_layer_init("conv6_1",conv6_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)



    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)

    # conv5_right=cnn.conv_layer_reuse("conv5",conv4_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    # conv6_right=cnn.conv_layer_reuse("conv6",conv5_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    # conv6_1_right=cnn.conv_layer_reuse("conv6_1",conv6_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    size_left=int(conv4_left.get_shape()[2])
    inner_product=[]
    for n in range(int(nclasses/4)):
        multiply=tf.multiply(conv4_left[:,:,0:size_left-n,:],conv4_right[:,:,n::,:])
        innerprod=tf.reduce_sum(multiply,-1)
        inner_product.append(tf.pad(innerprod,[[0,0],[0,0],[n,0]]))
    inner_product=tf.stack(inner_product)
    inner_product=tf.transpose(inner_product,[1,2,3,0])


    # shape=[1,1,160,160]
    # concat_features=cnn.conv_layer_init("concat_features",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,48,64]
    decod_conv3=cnn.conv_layer_init("decod_conv3",inner_product,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,128]
    decod_conv4=cnn.conv_layer_init("decod_conv4",decod_conv3,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,256]
    decod_conv4_2=cnn.conv_layer_init("decod_conv4_2",decod_conv4,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,256]
    decod_conv4_3=cnn.conv_layer_init("decod_conv4_3",decod_conv4_2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,w_init="xavier",training=training)

    deconv3=cnn.deconv_layer("deconv3",decod_conv4_3,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),128])
    deconv3=deconv3+decod_conv4

    deconv4=cnn.deconv_layer("deconv4",deconv3,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),64])
    deconv4=deconv4+decod_conv3

    shape=[3,3,64,1]
    final_score_2=cnn.conv_layer_init("final_score_2",deconv4,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    final_score_2=tf.image.resize_images(final_score_2,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    final_score_2=final_score_2#+final_score_1



    upsample2=cnn.deconv_layer("upsample2",deconv4,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    concat_features_3=tf.concat([upsample2,conv2_left,conv2_right],3)

    shape=[3,3,128,64]
    decod_conv5=cnn.conv_layer_init("decod_conv5",concat_features_3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,128]
    decod_conv6=cnn.conv_layer_init("decod_conv6",decod_conv5,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,256]
    decod_conv6_1=cnn.conv_layer_init("decod_conv6_1",decod_conv6,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,256]
    decod_conv6_2=cnn.conv_layer_init("decod_conv6_2",decod_conv6_1,shape,batch_norm=batch_norm,stride=[1,1],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv5=cnn.deconv_layer("deconv5",decod_conv6_2,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    deconv5=deconv5+decod_conv6

    deconv6=cnn.deconv_layer("deconv6",deconv5,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    deconv6=deconv6+decod_conv5

    shape=[3,3,64,1]
    final_score_3=cnn.conv_layer_init("final_score_3",deconv6,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    final_score_3=tf.image.resize_images(final_score_3,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    final_score_3=final_score_3#+final_score_2



    upsample3=cnn.deconv_layer("upsample3",deconv6,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    concat_features_4=tf.concat([upsample3,x_left,x_right],3)

    shape=[3,3,38,32]
    decod_conv7=cnn.conv_layer_init("decod_conv7",concat_features_4,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,64]
    decod_conv8=cnn.conv_layer_init("decod_conv8",decod_conv7,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,128]
    decod_conv8_1=cnn.conv_layer_init("decod_conv8_1",decod_conv8,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,128]
    decod_conv8_2=cnn.conv_layer_init("decod_conv8_2",decod_conv8_1,shape,batch_norm=batch_norm,stride=[1,1],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv7=cnn.deconv_layer("deconv7",decod_conv8_2,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    deconv7=deconv7+decod_conv8

    deconv8=cnn.deconv_layer("deconv8",deconv7,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    deconv8=deconv8+decod_conv7


    shape=[3,3,32,1]
    final_score=cnn.conv_layer_init("final_score",deconv8,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    #final_score_4=tf.image.resize_images(final_score_4,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #final_score_4=final_score_4#+final_score_3



    if training:
        loss_1=cnn.loss("loss_1", final_score_1,y,loss_type="regression")
        loss_2=cnn.loss("loss_2", final_score_2,y,loss_type="regression")
        loss_3=cnn.loss("loss_3", final_score_3,y,loss_type="regression")
        loss_7=cnn.loss("loss_7", final_score,y,loss_type="regression")

        #loss_r=cnn.loss("loss_r", final_refine,y,loss_type="regression")
        loss=loss_2+loss_3+loss_7

        return loss,final_score
    else:

        return final_score,final_score_1,final_score_2,final_score_3


def init_shallow_denser(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])
    y_expand=tf.expand_dims(y,-1)


    shape=[5,5,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,128]
    conv5_left=cnn.conv_layer_init("conv5",conv4_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    conv6_left=cnn.conv_layer_init("conv6",conv5_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    conv6_1_left=cnn.conv_layer_init("conv6_1",conv6_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)

    conv5_right=cnn.conv_layer_reuse("conv5",conv4_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv6_right=cnn.conv_layer_reuse("conv6",conv5_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv6_1_right=cnn.conv_layer_reuse("conv6_1",conv6_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)


    concat_features=tf.concat([conv6_1_left,conv6_1_right],3)#,conv5_right_strided_3,conv5_right_strided_7,conv5_right_strided_11],3)
    # shape=[1,1,160,160]
    # concat_features=cnn.conv_layer_init("concat_features",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    decod_conv1=cnn.conv_layer_init("decod_conv1",concat_features,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv1=cnn.conv_layer_init("decod_conv12",decod_conv1,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,512]
    decod_conv2=cnn.conv_layer_init("decod_conv2",decod_conv1,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,512]
    decod_conv2=cnn.conv_layer_init("decod_conv22",decod_conv2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,1024]
    decod_conv2_2=cnn.conv_layer_init("decod_conv2_2",decod_conv2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)

    shape=[3,3,1024,1024]
    decod_conv2_3=cnn.conv_layer_init("decod_conv2_3",decod_conv2_2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,w_init="xavier",training=training)

    deconv1=cnn.deconv_layer("deconv1",decod_conv2_3,512,filter_size=3,stride=2,output_shape=[batch_size,int(rows/16),int(cols/16),512])
    shape=[3,3,512,512]
    deconv1=cnn.conv_layer_init("deconv1_conv1",deconv1,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv1=deconv1+decod_conv2

    deconv2=cnn.deconv_layer("deconv2",deconv1,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),256])
    shape=[3,3,256,256]
    deconv2=cnn.conv_layer_init("deconv1_conv2",deconv2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv2=deconv2+decod_conv1

    shape=[3,3,256,1]
    final_score_1=cnn.conv_layer_init("final_score_1",deconv2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_1=tf.image.resize_images(final_score_1,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_1=tf.image.resize_images(y_expand,[int(rows/8),int(cols/8)])/8
    #y_1=tf.squeeze(y_1,-1)

    upsample1=cnn.deconv_layer("upsample1",deconv2,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    concat_features_2=tf.concat([upsample1,conv4_left,conv4_right],3)

    shape=[3,3,256,128]
    decod_conv3=cnn.conv_layer_init("decod_conv3",concat_features_2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    decod_conv3=cnn.conv_layer_init("decod_conv32",decod_conv3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,256]
    decod_conv4=cnn.conv_layer_init("decod_conv4",decod_conv3,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    decod_conv4=cnn.conv_layer_init("decod_conv42",decod_conv4,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,512]
    decod_conv4_2=cnn.conv_layer_init("decod_conv4_2",decod_conv4,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)

    shape=[3,3,512,512]
    decod_conv4_3=cnn.conv_layer_init("decod_conv4_3",decod_conv4_2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,w_init="xavier",training=training)

    deconv3=cnn.deconv_layer("deconv3",decod_conv4_3,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),256])
    shape=[3,3,256,256]
    deconv3=cnn.conv_layer_init("deconv3_conv11",deconv3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv3=deconv3+decod_conv4

    deconv4=cnn.deconv_layer("deconv4",deconv3,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    shape=[3,3,128,128]
    deconv4=cnn.conv_layer_init("deconv4_conv11",deconv4,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv4=deconv4+decod_conv3

    shape=[3,3,128,1]
    final_score_2=cnn.conv_layer_init("final_score_2",deconv4,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    #
    final_score_2=tf.image.resize_images(final_score_2,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #final_score_2=final_score_2#+final_score_1
    #y_2=tf.image.resize_images(y_expand,[int(rows/4),int(cols/4)])/4
    #y_2=tf.squeeze(y_2,-1)


    upsample2=cnn.deconv_layer("upsample2",deconv4,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    concat_features_3=tf.concat([upsample2,conv2_left,conv2_right],3)

    shape=[3,3,128,64]
    decod_conv5=cnn.conv_layer_init("decod_conv5",concat_features_3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    decod_conv5=cnn.conv_layer_init("decod_conv52",decod_conv5,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,128]
    decod_conv6=cnn.conv_layer_init("decod_conv6",decod_conv5,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    decod_conv6=cnn.conv_layer_init("decod_conv62",decod_conv6,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,256]
    decod_conv6_1=cnn.conv_layer_init("decod_conv6_1",decod_conv6,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,256]
    decod_conv6_2=cnn.conv_layer_init("decod_conv6_2",decod_conv6_1,shape,batch_norm=batch_norm,stride=[1,1],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv5=cnn.deconv_layer("deconv5",decod_conv6_2,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    shape=[3,3,128,128]
    deconv5=cnn.conv_layer_init("deconv5_conv1",deconv5,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv5=deconv5+decod_conv6

    deconv6=cnn.deconv_layer("deconv6",deconv5,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    shape=[3,3,64,64]
    deconv6=cnn.conv_layer_init("deconv6_conv1",deconv6,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv6=deconv6+decod_conv5

    shape=[3,3,64,1]
    final_score_3=cnn.conv_layer_init("final_score_3",deconv6,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    #
    final_score_3=tf.image.resize_images(final_score_3,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #final_score_3=final_score_3#+final_score_2
    #y_3=tf.image.resize_images(y_expand,[int(rows/2),int(cols/2)])/2
    #y_3=tf.squeeze(y_3,-1)


    upsample3=cnn.deconv_layer("upsample3",deconv6,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    concat_features_4=tf.concat([upsample3,x_left,x_right],3)

    shape=[3,3,38,32]
    decod_conv7=cnn.conv_layer_init("decod_conv7",concat_features_4,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    decod_conv7=cnn.conv_layer_init("decod_conv72",decod_conv7,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,32,64]
    decod_conv8=cnn.conv_layer_init("decod_conv8",decod_conv7,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    decod_conv8=cnn.conv_layer_init("decod_conv82",decod_conv8,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,128]
    decod_conv8_1=cnn.conv_layer_init("decod_conv8_1",decod_conv8,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,128]
    decod_conv8_2=cnn.conv_layer_init("decod_conv8_2",decod_conv8_1,shape,batch_norm=batch_norm,stride=[1,1],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv7=cnn.deconv_layer("deconv7",decod_conv8_2,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    shape=[3,3,64,64]
    deconv7=cnn.conv_layer_init("deconv7_conv1",deconv7,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv7=deconv7+decod_conv8

    deconv8=cnn.deconv_layer("deconv8",deconv7,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    shape=[3,3,32,32]
    deconv8=cnn.conv_layer_init("deconv82",deconv8,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv8=deconv8+decod_conv7


    shape=[3,3,32,1]
    final_score=cnn.conv_layer_init("final_score",deconv8,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    #final_score_4=tf.image.resize_images(final_score_4,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #final_score_4=final_score_4#+final_score_3



    if training:
        loss_1=cnn.loss("loss_1", final_score_1,y/8,loss_type="regression")
        loss_2=cnn.loss("loss_2", final_score_2,y/4,loss_type="regression")
        loss_3=cnn.loss("loss_3", final_score_3,y/2,loss_type="regression")
        loss_7=cnn.loss("loss_7", final_score,y,loss_type="regression")

        #loss_r=cnn.loss("loss_r", final_refine,y,loss_type="regression")
        loss=loss_3+loss_7+loss_2+loss_1

        return loss,final_score
    else:

        return final_score,final_score_1,final_score_2,final_score_3


def init_shallow(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])
    y_expand=tf.expand_dims(y,-1)


    shape=[5,5,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,128]
    conv5_left=cnn.conv_layer_init("conv5",conv4_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    conv6_left=cnn.conv_layer_init("conv6",conv5_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    conv6_1_left=cnn.conv_layer_init("conv6_1",conv6_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)

    conv5_right=cnn.conv_layer_reuse("conv5",conv4_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv6_right=cnn.conv_layer_reuse("conv6",conv5_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv6_1_right=cnn.conv_layer_reuse("conv6_1",conv6_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)


    concat_features=tf.concat([conv6_1_left,conv6_1_right],3)#,conv5_right_strided_3,conv5_right_strided_7,conv5_right_strided_11],3)
    # shape=[1,1,160,160]
    # concat_features=cnn.conv_layer_init("concat_features",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    decod_conv1=cnn.conv_layer_init("decod_conv1",concat_features,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,512]
    decod_conv2=cnn.conv_layer_init("decod_conv2",decod_conv1,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,1024]
    decod_conv2_2=cnn.conv_layer_init("decod_conv2_2",decod_conv2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)

    shape=[3,3,1024,1024]
    decod_conv2_3=cnn.conv_layer_init("decod_conv2_3",decod_conv2_2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,w_init="xavier",training=training)

    deconv1=cnn.deconv_layer("deconv1",decod_conv2_3,512,filter_size=3,stride=2,output_shape=[batch_size,int(rows/16),int(cols/16),512])
    deconv1=deconv1+decod_conv2

    deconv2=cnn.deconv_layer("deconv2",deconv1,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),256])
    deconv2=deconv2+decod_conv1

    shape=[3,3,256,1]
    final_score_1=cnn.conv_layer_init("final_score_1",deconv2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_1=tf.image.resize_images(final_score_1,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_1=tf.image.resize_images(y_expand,[int(rows/8),int(cols/8)])/8
    #y_1=tf.squeeze(y_1,-1)

    upsample1=cnn.deconv_layer("upsample1",deconv2,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    concat_features_2=tf.concat([upsample1,conv4_left,conv4_right],3)

    shape=[3,3,256,128]
    decod_conv3=cnn.conv_layer_init("decod_conv3",concat_features_2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,256]
    decod_conv4=cnn.conv_layer_init("decod_conv4",decod_conv3,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,512]
    decod_conv4_2=cnn.conv_layer_init("decod_conv4_2",decod_conv4,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)

    shape=[3,3,512,512]
    decod_conv4_3=cnn.conv_layer_init("decod_conv4_3",decod_conv4_2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,w_init="xavier",training=training)

    deconv3=cnn.deconv_layer("deconv3",decod_conv4_3,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),256])
    deconv3=deconv3+decod_conv4

    deconv4=cnn.deconv_layer("deconv4",deconv3,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    deconv4=deconv4+decod_conv3

    shape=[3,3,128,1]
    final_score_2=cnn.conv_layer_init("final_score_2",deconv4,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    #
    final_score_2=tf.image.resize_images(final_score_2,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #final_score_2=final_score_2#+final_score_1
    #y_2=tf.image.resize_images(y_expand,[int(rows/4),int(cols/4)])/4
    #y_2=tf.squeeze(y_2,-1)


    upsample2=cnn.deconv_layer("upsample2",deconv4,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    concat_features_3=tf.concat([upsample2,conv2_left,conv2_right],3)

    shape=[3,3,128,64]
    decod_conv5=cnn.conv_layer_init("decod_conv5",concat_features_3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,128]
    decod_conv6=cnn.conv_layer_init("decod_conv6",decod_conv5,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,256]
    decod_conv6_1=cnn.conv_layer_init("decod_conv6_1",decod_conv6,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,256]
    decod_conv6_2=cnn.conv_layer_init("decod_conv6_2",decod_conv6_1,shape,batch_norm=batch_norm,stride=[1,1],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv5=cnn.deconv_layer("deconv5",decod_conv6_2,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    deconv5=deconv5+decod_conv6

    deconv6=cnn.deconv_layer("deconv6",deconv5,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    deconv6=deconv6+decod_conv5

    shape=[3,3,64,1]
    final_score_3=cnn.conv_layer_init("final_score_3",deconv6,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    #
    final_score_3=tf.image.resize_images(final_score_3,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #final_score_3=final_score_3#+final_score_2
    #y_3=tf.image.resize_images(y_expand,[int(rows/2),int(cols/2)])/2
    #y_3=tf.squeeze(y_3,-1)


    upsample3=cnn.deconv_layer("upsample3",deconv6,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    concat_features_4=tf.concat([upsample3,x_left,x_right],3)

    shape=[3,3,38,32]
    decod_conv7=cnn.conv_layer_init("decod_conv7",concat_features_4,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,64]
    decod_conv8=cnn.conv_layer_init("decod_conv8",decod_conv7,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,128]
    decod_conv8_1=cnn.conv_layer_init("decod_conv8_1",decod_conv8,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,128]
    decod_conv8_2=cnn.conv_layer_init("decod_conv8_2",decod_conv8_1,shape,batch_norm=batch_norm,stride=[1,1],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv7=cnn.deconv_layer("deconv7",decod_conv8_2,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    deconv7=deconv7+decod_conv8

    deconv8=cnn.deconv_layer("deconv8",deconv7,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    deconv8=deconv8+decod_conv7


    shape=[3,3,32,1]
    final_score=cnn.conv_layer_init("final_score",deconv8,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    #final_score_4=tf.image.resize_images(final_score_4,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #final_score_4=final_score_4#+final_score_3



    if training:
        loss_1=cnn.loss("loss_1", final_score_1,y/8,loss_type="regression")
        loss_2=cnn.loss("loss_2", final_score_2,y/4,loss_type="regression")
        loss_3=cnn.loss("loss_3", final_score_3,y/2,loss_type="regression")
        loss_7=cnn.loss("loss_7", final_score,y,loss_type="regression")

        #loss_r=cnn.loss("loss_r", final_refine,y,loss_type="regression")
        loss=loss_7+loss_2+loss_1+loss_3

        return loss,final_score
    else:

        return final_score,final_score_1,final_score_2,final_score_3


def init_shallow_allfeatures(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])


    shape=[5,5,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,128]
    conv5_left=cnn.conv_layer_init("conv5",conv4_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    conv6_left=cnn.conv_layer_init("conv6",conv5_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    conv6_1_left=cnn.conv_layer_init("conv6_1",conv6_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)

    conv5_right=cnn.conv_layer_reuse("conv5",conv4_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv6_right=cnn.conv_layer_reuse("conv6",conv5_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv6_1_right=cnn.conv_layer_reuse("conv6_1",conv6_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)


    conv2_left_r8=tf.image.resize_images(conv2_left,[int(rows/8),int(cols/8)])
    conv2_right_r8=tf.image.resize_images(conv2_right,[int(rows/8),int(cols/8)])
    conv4_left_r8=tf.image.resize_images(conv4_left,[int(rows/8),int(cols/8)])
    conv4_right_r8=tf.image.resize_images(conv4_right,[int(rows/8),int(cols/8)])
    concat_features=tf.concat([conv6_1_left,conv6_1_right,conv2_left_r8,conv2_right_r8,conv4_left_r8,conv4_right_r8],3)#,conv5_right_strided_3,conv5_right_strided_7,conv5_right_strided_11],3)
    shape=[1,1,448,256]
    concat_features=cnn.conv_layer_init("compress_1",concat_features,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    # shape=[1,1,160,160]
    # concat_features=cnn.conv_layer_init("concat_features",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    decod_conv1=cnn.conv_layer_init("decod_conv1",concat_features,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,512]
    decod_conv2=cnn.conv_layer_init("decod_conv2",decod_conv1,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,1024]
    decod_conv2_2=cnn.conv_layer_init("decod_conv2_2",decod_conv2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)

    shape=[3,3,1024,1024]
    decod_conv2_3=cnn.conv_layer_init("decod_conv2_3",decod_conv2_2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,w_init="xavier",training=training)

    deconv1=cnn.deconv_layer("deconv1",decod_conv2_3,512,filter_size=3,stride=2,output_shape=[batch_size,int(rows/16),int(cols/16),512])
    deconv1=deconv1+decod_conv2

    deconv2=cnn.deconv_layer("deconv2",deconv1,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),256])
    deconv2=deconv2+decod_conv1

    shape=[3,3,256,1]
    final_score_1=cnn.conv_layer_init("final_score_1",deconv2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_1=tf.image.resize_images(final_score_1,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    upsample1=cnn.deconv_layer("upsample1",deconv2,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])

    conv2_left_r4=tf.image.resize_images(conv2_left,[int(rows/4),int(cols/4)])
    conv2_right_r4=tf.image.resize_images(conv2_right,[int(rows/4),int(cols/4)])
    conv6_1_left_r4=tf.image.resize_images(conv6_1_left,[int(rows/4),int(cols/4)])
    conv6_1_right_r4=tf.image.resize_images(conv6_1_right,[int(rows/4),int(cols/4)])
    concat_features_2=tf.concat([upsample1,conv4_left,conv4_right,conv2_left_r4,conv2_right_r4,conv6_1_left_r4,conv6_1_right_r4],3)
    shape=[1,1,576,256]
    concat_features_2=cnn.conv_layer_init("compress_2",concat_features_2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    shape=[3,3,256,128]
    decod_conv3=cnn.conv_layer_init("decod_conv3",concat_features_2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,256]
    decod_conv4=cnn.conv_layer_init("decod_conv4",decod_conv3,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,512]
    decod_conv4_2=cnn.conv_layer_init("decod_conv4_2",decod_conv4,shape,batch_norm=batch_norm,activation=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)

    shape=[3,3,512,512]
    decod_conv4_3=cnn.conv_layer_init("decod_conv4_3",decod_conv4_2,shape,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,w_init="xavier",training=training)

    deconv3=cnn.deconv_layer("deconv3",decod_conv4_3,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),256])
    deconv3=deconv3+decod_conv4

    deconv4=cnn.deconv_layer("deconv4",deconv3,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    deconv4=deconv4+decod_conv3

    shape=[3,3,128,1]
    final_score_2=cnn.conv_layer_init("final_score_2",deconv4,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    final_score_2=tf.image.resize_images(final_score_2,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    final_score_2=final_score_2#+final_score_1



    upsample2=cnn.deconv_layer("upsample2",deconv4,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])

    conv4_left_r2=tf.image.resize_images(conv4_left,[int(rows/2),int(cols/2)])
    conv4_right_r2=tf.image.resize_images(conv4_right,[int(rows/2),int(cols/2)])
    conv6_1_left_r2=tf.image.resize_images(conv6_1_left,[int(rows/2),int(cols/2)])
    conv6_1_right_r2=tf.image.resize_images(conv6_1_right,[int(rows/2),int(cols/2)])
    concat_features_3=tf.concat([upsample2,conv2_left,conv2_right,conv4_left_r2,conv4_right_r2,conv6_1_left_r2,conv6_1_right_r2],3)
    shape=[1,1,512,128]
    concat_features_3=cnn.conv_layer_init("compress_3",concat_features_3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    shape=[3,3,128,64]
    decod_conv5=cnn.conv_layer_init("decod_conv5",concat_features_3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,128]
    decod_conv6=cnn.conv_layer_init("decod_conv6",decod_conv5,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,256]
    decod_conv6_1=cnn.conv_layer_init("decod_conv6_1",decod_conv6,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,256]
    decod_conv6_2=cnn.conv_layer_init("decod_conv6_2",decod_conv6_1,shape,batch_norm=batch_norm,stride=[1,1],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv5=cnn.deconv_layer("deconv5",decod_conv6_2,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    deconv5=deconv5+decod_conv6

    deconv6=cnn.deconv_layer("deconv6",deconv5,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    deconv6=deconv6+decod_conv5

    shape=[3,3,64,1]
    final_score_3=cnn.conv_layer_init("final_score_3",deconv6,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    final_score_3=tf.image.resize_images(final_score_3,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    final_score_3=final_score_3#+final_score_2



    upsample3=cnn.deconv_layer("upsample3",deconv6,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    concat_features_4=tf.concat([upsample3,x_left,x_right],3)

    shape=[3,3,38,32]
    decod_conv7=cnn.conv_layer_init("decod_conv7",concat_features_4,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,64]
    decod_conv8=cnn.conv_layer_init("decod_conv8",decod_conv7,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,128]
    decod_conv8_1=cnn.conv_layer_init("decod_conv8_1",decod_conv8,shape,batch_norm=batch_norm,stride=[2,2],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,128]
    decod_conv8_2=cnn.conv_layer_init("decod_conv8_2",decod_conv8_1,shape,batch_norm=batch_norm,stride=[1,1],activation=batch_norm,padding=padding,w_init="xavier",training=training)

    deconv7=cnn.deconv_layer("deconv7",decod_conv8_2,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    deconv7=deconv7+decod_conv8

    deconv8=cnn.deconv_layer("deconv8",deconv7,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    deconv8=deconv8+decod_conv7


    shape=[3,3,32,1]
    final_score=cnn.conv_layer_init("final_score",deconv8,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    #final_score_4=tf.image.resize_images(final_score_4,[int(rows),int(cols)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #final_score_4=final_score_4#+final_score_3



    if training:
        loss_1=cnn.loss("loss_1", final_score_1,y,loss_type="regression")
        loss_2=cnn.loss("loss_2", final_score_2,y,loss_type="regression")
        loss_3=cnn.loss("loss_3", final_score_3,y,loss_type="regression")
        loss_7=cnn.loss("loss_7", final_score,y,loss_type="regression")

        #loss_r=cnn.loss("loss_r", final_refine,y,loss_type="regression")
        loss=loss_1+loss_2+loss_3+loss_7

        return loss,final_score
    else:

        return final_score,final_score_1,final_score_2,final_score_3


def init_pyramid_residual(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])
    y_expand=tf.expand_dims(y,-1)

    shape=[3,3,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    print("conv2_left",conv2_left.get_shape())
    shape=[3,3,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,128]
    conv5_left=cnn.conv_layer_init("conv5",conv4_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    conv6_left=cnn.conv_layer_init("conv6",conv5_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    conv6_1_left=cnn.conv_layer_init("conv6_1",conv6_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)


    shape=[3,3,128,256]
    conv7_left=cnn.conv_layer_init("conv7",conv6_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    conv8_left=cnn.conv_layer_init("conv8",conv7_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    conv8_1_left=cnn.conv_layer_init("conv8_1",conv8_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,512]
    conv9_left=cnn.conv_layer_init("conv9",conv8_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,512]
    conv10_left=cnn.conv_layer_init("conv10",conv9_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    conv10_1_left=cnn.conv_layer_init("conv10_1",conv10_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)

    # shape=[5,5,512,1024]
    # conv11_left=cnn.conv_layer_init("conv11",conv10_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    # shape=[3,3,1024,1024]
    # conv12_left=cnn.conv_layer_init("conv12",conv11_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    # conv12_1_left=cnn.conv_layer_init("conv12_1",conv12_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)

    conv5_right=cnn.conv_layer_reuse("conv5",conv4_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv6_right=cnn.conv_layer_reuse("conv6",conv5_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv6_1_right=cnn.conv_layer_reuse("conv6_1",conv6_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv7_right=cnn.conv_layer_reuse("conv7",conv6_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv8_right=cnn.conv_layer_reuse("conv8",conv7_right,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,training=training)
    conv8_1_right=cnn.conv_layer_reuse("conv8_1",conv8_right,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,training=training)

    conv9_right=cnn.conv_layer_reuse("conv9",conv8_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv10_right=cnn.conv_layer_reuse("conv10",conv9_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv10_1_right=cnn.conv_layer_reuse("conv10_1",conv10_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)

    left_c=int(conv10_1_right.get_shape()[2])


    # conv11_right=cnn.conv_layer_reuse("conv11",conv10_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    # conv12_right=cnn.conv_layer_reuse("conv12",conv11_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    # conv12_1_right=cnn.conv_layer_reuse("conv12_1",conv12_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    # shape=conv10_1_right.get_shape()
    # classes_down=int(192/32)
    # concat_features=[]
    # for i in range(classes_down):
    #     mult=tf.multiply(conv10_1_left[:,:,i::,:],conv10_1_right[:,:,0:shape[2]-i,:])
    #     innerprod=tf.reduce_sum(mult,-1)
    #     concat_features.append(tf.pad(innerprod,[[0,0],[0,0], [i,0]]))
    #
    # concat_features=tf.transpose(tf.stack(concat_features),perm=[1,2,3,0])




    concat_features=tf.concat([conv10_1_left,conv10_1_right],3)#,conv5_right_strided_3,conv5_right_strided_7,conv5_right_strided_11],3)
    #concat_features=tf.concat([conv10_1_left,conv10_1_right,conv10_1_right_strided_2,conv10_1_right_strided_3],3)#,conv5_right_strided_3,conv5_right_strided_7,conv5_right_strided_11],3)
    #shape=[1,1,2048,1024]
    #concat_features=cnn.conv_layer_init("reduce_feat",concat_features,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    # shape=[1,1,160,160]
    # concat_features=cnn.conv_layer_init("concat_features",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,1024,1024]
    decod_conv1=cnn.conv_layer_init("decod_conv1",concat_features,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,1024,1024]
    decod_conv2=cnn.conv_layer_init("decod_conv2",decod_conv1,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv2_2=cnn.conv_layer_init("decod_conv2_2",decod_conv2,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,1024,1]
    final_score_1=cnn.conv_layer_init("final_score_1",decod_conv2_2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_1=tf.image.resize_images(final_score_1,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_1=tf.image.resize_images(y_expand,[int(rows/32),int(cols/32)]),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_1=tf.squeeze(y_1,-1)

    deconv1=cnn.deconv_layer("deconv1",decod_conv2_2,512,filter_size=3,stride=2,output_shape=[batch_size,int(rows/16),int(cols/16),512])
    concat_features_2=tf.concat([deconv1,conv8_1_left,conv8_1_right],3)

    shape=[3,3,1024,512]
    decod_conv3=cnn.conv_layer_init("decod_conv3",concat_features_2,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,512]
    decod_conv4=cnn.conv_layer_init("decod_conv4",decod_conv3,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv4_2=cnn.conv_layer_init("decod_conv4_2",decod_conv4,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,512,1]
    final_score_2=cnn.conv_layer_init("final_score_2",decod_conv4_2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    final_score_2=tf.image.resize_images(final_score_2,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_2=tf.image.resize_images(y_expand,[int(rows/16),int(cols/16)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_2=tf.squeeze(y_2,-1)



    deconv2=cnn.deconv_layer("deconv2",decod_conv4_2,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),256])
    concat_features_3=tf.concat([deconv2,conv6_1_left,conv6_1_right],3)

    shape=[3,3,512,256]
    decod_conv5=cnn.conv_layer_init("decod_conv5",concat_features_3,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    decod_conv6=cnn.conv_layer_init("decod_conv6",decod_conv5,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv6_1=cnn.conv_layer_init("decod_conv6_1",decod_conv6,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,1]
    final_score_3=cnn.conv_layer_init("final_score_3",decod_conv6_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_3=tf.image.resize_images(final_score_3,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_3=tf.image.resize_images(y_expand,[int(rows/8),int(cols/8)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_3=tf.squeeze(y_3,-1)


    print("oiiiiiiiii",decod_conv6_1.get_shape())
    deconv3=cnn.deconv_layer("deconv3",decod_conv6_1,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    concat_features_4=tf.concat([deconv3,conv4_left,conv4_right],3)

    shape=[3,3,256,128]
    decod_conv7=cnn.conv_layer_init("decod_conv7",concat_features_4,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    decod_conv8=cnn.conv_layer_init("decod_conv8",decod_conv7,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv8_1=cnn.conv_layer_init("decod_conv8_1",decod_conv8,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,1]
    final_score_4=cnn.conv_layer_init("final_score_4",decod_conv8_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_4=tf.image.resize_images(final_score_4,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_4=tf.image.resize_images(y_expand,[int(rows/4),int(cols/4)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_4=tf.squeeze(y_4,-1)



    deconv4=cnn.deconv_layer("deconv4",decod_conv8_1,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    concat_features_5=tf.concat([deconv4,conv2_left,conv2_right],3)

    shape=[3,3,128,64]
    decod_conv9=cnn.conv_layer_init("decod_conv9",concat_features_5,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    decod_conv10=cnn.conv_layer_init("decod_conv10",decod_conv9,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv10_1=cnn.conv_layer_init("decod_conv10_1",decod_conv10,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)


    shape=[3,3,64,1]
    final_score_5=cnn.conv_layer_init("final_score_5",decod_conv10_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_5=tf.image.resize_images(final_score_5,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_5=tf.image.resize_images(y_expand,[int(rows/2),int(cols/2)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_5=tf.squeeze(y_5,-1)


    deconv5=cnn.deconv_layer("deconv5",decod_conv10_1,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    concat_features_6=tf.concat([deconv5,x_left,x_right],3)

    shape=[3,3,38,32]
    decod_conv11=cnn.conv_layer_init("decod_conv11",concat_features_6,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    decod_conv12=cnn.conv_layer_init("decod_conv12",decod_conv11,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv13=cnn.conv_layer_init("decod_conv13",decod_conv12,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,32,1]
    final_score=cnn.conv_layer_init("final_score_6",decod_conv13,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    #final_score_6=tf.image.resize_images(final_score,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #final_score=final_score_6#+final_score_5


    # concat_preds=tf.concat([final_score_1,final_score_2,final_score_3,final_score_4,final_score_5,final_score_6],3)
    # shape=[3,3,6,32]
    # concat_preds_conv=cnn.conv_layer_init("concat_preds_conv",concat_preds,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    # shape=[3,3,32,32]
    # concat_preds_conv2=cnn.conv_layer_init("concat_preds_conv2",concat_preds_conv,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    # shape=[3,3,32,1]
    # final_score=cnn.conv_layer_init("final_score",concat_preds_conv2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)


    if training:
        loss_1=cnn.loss("loss_1", final_score_1,y,loss_type="regression")
        loss_2=cnn.loss("loss_2", final_score_2,y,loss_type="regression")
        loss_3=cnn.loss("loss_3", final_score_3,y,loss_type="regression")
        loss_4=cnn.loss("loss_4", final_score_4,y,loss_type="regression")
        loss_5=cnn.loss("loss_5", final_score_5,y,loss_type="regression")
        #loss_6=cnn.loss("loss_6", final_score_6,y,loss_type="regression")
        loss_7=cnn.loss("loss_7", final_score,y,loss_type="regression")

        #loss_r=cnn.loss("loss_r", final_refine,y,loss_type="regression")
        loss=loss_7#loss_1+loss_2+loss_3+loss_4+loss_5+loss_7#+loss_6

        return loss,final_score
    else:

        return final_score,final_score_1,final_score_2,final_score_3



def init_pyramid_residual_3strided(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])
    y_expand=tf.expand_dims(y,-1)

    shape=[7,7,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[3,3],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    print("conv2_left",conv2_left.get_shape())
    shape=[7,7,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[3,3],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,128]
    conv5_left=cnn.conv_layer_init("conv5",conv4_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    conv6_left=cnn.conv_layer_init("conv6",conv5_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    conv6_1_left=cnn.conv_layer_init("conv6_1",conv6_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)


    shape=[3,3,128,256]
    conv7_left=cnn.conv_layer_init("conv7",conv6_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    conv8_left=cnn.conv_layer_init("conv8",conv7_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    conv8_1_left=cnn.conv_layer_init("conv8_1",conv8_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,512]
    conv9_left=cnn.conv_layer_init("conv9",conv8_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,512]
    conv10_left=cnn.conv_layer_init("conv10",conv9_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    conv10_1_left=cnn.conv_layer_init("conv10_1",conv10_left,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[3,3],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[3,3],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)

    conv5_right=cnn.conv_layer_reuse("conv5",conv4_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv6_right=cnn.conv_layer_reuse("conv6",conv5_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv6_1_right=cnn.conv_layer_reuse("conv6_1",conv6_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv7_right=cnn.conv_layer_reuse("conv7",conv6_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv8_right=cnn.conv_layer_reuse("conv8",conv7_right,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,training=training)
    conv8_1_right=cnn.conv_layer_reuse("conv8_1",conv8_right,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,training=training)

    conv9_right=cnn.conv_layer_reuse("conv9",conv8_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv10_right=cnn.conv_layer_reuse("conv10",conv9_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv10_1_right=cnn.conv_layer_reuse("conv10_1",conv10_right,batch_norm=False,activation=False,stride=[1,1],padding=padding,training=training)

    left_c=int(conv10_1_right.get_shape()[2])





    concat_features=tf.concat([conv10_1_left,conv10_1_right],3)#,conv5_right_strided_3,conv5_right_strided_7,conv5_right_strided_11],3)

    shape=[3,3,1024,1024]
    decod_conv1=cnn.conv_layer_init("decod_conv1",concat_features,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,1024,1024]
    decod_conv2=cnn.conv_layer_init("decod_conv2",decod_conv1,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv2_2=cnn.conv_layer_init("decod_conv2_2",decod_conv2,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,1024,1]
    final_score_1=cnn.conv_layer_init("final_score_1",decod_conv2_2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_1=tf.image.resize_images(final_score_1,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_1=tf.image.resize_images(y_expand,[int(rows/32),int(cols/32)]),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_1=tf.squeeze(y_1,-1)

    deconv1=cnn.deconv_layer("deconv1",decod_conv2_2,512,filter_size=3,stride=2,output_shape=[batch_size,int(rows/36),int(cols/36),512])
    concat_features_2=tf.concat([deconv1,conv8_1_left,conv8_1_right],3)

    shape=[3,3,1024,512]
    decod_conv3=cnn.conv_layer_init("decod_conv3",concat_features_2,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,512]
    decod_conv4=cnn.conv_layer_init("decod_conv4",decod_conv3,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv4_2=cnn.conv_layer_init("decod_conv4_2",decod_conv4,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,512,1]
    final_score_2=cnn.conv_layer_init("final_score_2",decod_conv4_2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    final_score_2=tf.image.resize_images(final_score_2,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_2=tf.image.resize_images(y_expand,[int(rows/16),int(cols/16)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_2=tf.squeeze(y_2,-1)



    deconv2=cnn.deconv_layer("deconv2",decod_conv4_2,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/18),int(cols/18),256])
    concat_features_3=tf.concat([deconv2,conv6_1_left,conv6_1_right],3)

    shape=[3,3,512,256]
    decod_conv5=cnn.conv_layer_init("decod_conv5",concat_features_3,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    decod_conv6=cnn.conv_layer_init("decod_conv6",decod_conv5,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv6_1=cnn.conv_layer_init("decod_conv6_1",decod_conv6,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,1]
    final_score_3=cnn.conv_layer_init("final_score_3",decod_conv6_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_3=tf.image.resize_images(final_score_3,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_3=tf.image.resize_images(y_expand,[int(rows/8),int(cols/8)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_3=tf.squeeze(y_3,-1)


    deconv3=cnn.deconv_layer("deconv3",decod_conv6_1,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/9),int(cols/9),128])
    concat_features_4=tf.concat([deconv3,conv4_left,conv4_right],3)

    shape=[3,3,256,128]
    decod_conv7=cnn.conv_layer_init("decod_conv7",concat_features_4,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    decod_conv8=cnn.conv_layer_init("decod_conv8",decod_conv7,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv8_1=cnn.conv_layer_init("decod_conv8_1",decod_conv8,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,1]
    final_score_4=cnn.conv_layer_init("final_score_4",decod_conv8_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_4=tf.image.resize_images(final_score_4,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_4=tf.image.resize_images(y_expand,[int(rows/4),int(cols/4)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_4=tf.squeeze(y_4,-1)

    deconv4=cnn.deconv_layer("deconv4",decod_conv8_1,64,filter_size=7,stride=3,output_shape=[batch_size,int(rows/3),int(cols/3),64])
    concat_features_5=tf.concat([deconv4,conv2_left,conv2_right],3)

    shape=[3,3,128,64]
    decod_conv9=cnn.conv_layer_init("decod_conv9",concat_features_5,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    decod_conv10=cnn.conv_layer_init("decod_conv10",decod_conv9,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv10_1=cnn.conv_layer_init("decod_conv10_1",decod_conv10,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)


    shape=[3,3,64,1]
    final_score_5=cnn.conv_layer_init("final_score_5",decod_conv10_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_5=tf.image.resize_images(final_score_5,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #y_5=tf.image.resize_images(y_expand,[int(rows/2),int(cols/2)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y_5=tf.squeeze(y_5,-1)


    deconv5=cnn.deconv_layer("deconv5",decod_conv10_1,32,filter_size=7,stride=3,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    concat_features_6=tf.concat([deconv5,x_left,x_right],3)

    shape=[3,3,38,32]
    decod_conv11=cnn.conv_layer_init("decod_conv11",concat_features_6,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    decod_conv12=cnn.conv_layer_init("decod_conv12",decod_conv11,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    decod_conv13=cnn.conv_layer_init("decod_conv13",decod_conv12,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,32,1]
    final_score=cnn.conv_layer_init("final_score_6",decod_conv13,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    #final_score_6=tf.image.resize_images(final_score,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #final_score=final_score_6#+final_score_5

    # concat_scores=tf.concat([final_score,x_left],3)
    # #
    # shape=[3,3,4,32]
    # refine1=cnn.conv_layer_init("refine1",concat_scores,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    # shape=[3,3,32,64]
    # refine2=cnn.conv_layer_init("refine2",refine1,shape,batch_norm=batch_norm,activation=True,stride=[2,2],padding=padding,w_init="xavier",training=training)
    # shape=[3,3,64,64]
    # refine3=cnn.conv_layer_init("refine3",refine2,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    #
    # shape=[3,3,64,128]
    # refine4=cnn.conv_layer_init("refine4",refine3,shape,batch_norm=batch_norm,activation=True,stride=[2,2],padding=padding,w_init="xavier",training=training)
    # shape=[3,3,128,128]
    # refine5=cnn.conv_layer_init("refine5",refine4,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    #
    # shape=[1,1,128,128]
    # refine_fc=cnn.conv_layer_init("refine_fc",refine5,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    #
    # deconv_refine1=cnn.deconv_layer("deconv_refine1",refine_fc,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    # deconv_refine1=deconv_refine1+refine3
    # shape=[3,3,64,64]
    # deconv_conv_refine1=cnn.conv_layer_init("deconv_conv_refine1",deconv_refine1,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    # deconv_conv_refine2=cnn.conv_layer_init("deconv_conv_refine2",deconv_conv_refine1,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    #
    # deconv_refine2=cnn.deconv_layer("deconv_refine2",deconv_conv_refine2,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    # deconv_refine2=deconv_refine2+refine1
    # shape=[3,3,32,32]
    # deconv2_conv_refine1=cnn.conv_layer_init("deconv2_conv_refine1",deconv_refine2,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    # deconv2_conv_refine2=cnn.conv_layer_init("deconv2_conv_refine2",deconv2_conv_refine1,shape,batch_norm=batch_norm,activation=True,padding=padding,w_init="xavier",training=training)
    #
    #
    # shape=[3,3,32,1]
    # final_refine=cnn.conv_layer_init("final_refine",deconv2_conv_refine2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    if training:
        loss_1=cnn.loss("loss_1", final_score_1,y,loss_type="regression")
        loss_2=cnn.loss("loss_2", final_score_2,y,loss_type="regression")
        loss_3=cnn.loss("loss_3", final_score_3,y,loss_type="regression")
        loss_4=cnn.loss("loss_4", final_score_4,y,loss_type="regression")
        loss_5=cnn.loss("loss_5", final_score_5,y,loss_type="regression")

        #loss_final_refine=cnn.loss("loss_final_refine", final_refine,y,loss_type="regression")
        loss_7=cnn.loss("loss_7", final_score,y,loss_type="regression")

        #loss_r=cnn.loss("loss_r", final_refine,y,loss_type="regression")
        loss=loss_1+loss_2+loss_3+loss_4+loss_5+loss_7#+loss_final_refine

        return loss,final_score
    else:

        return final_score,final_score_1,final_score_2,final_score_3


def init_pyramid_deeper(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])


    shape=[3,3,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    print("conv2_left",conv2_left.get_shape())
    shape=[3,3,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,128]
    conv5_left=cnn.conv_layer_init("conv5",conv4_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    conv6_left=cnn.conv_layer_init("conv6",conv5_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    conv6_1_left=cnn.conv_layer_init("conv6_1",conv6_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    shape=[3,3,128,256]
    conv7_left=cnn.conv_layer_init("conv7",conv6_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    conv8_left=cnn.conv_layer_init("conv8",conv7_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    conv8_1_left=cnn.conv_layer_init("conv8_1",conv8_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,512]
    conv9_left=cnn.conv_layer_init("conv9",conv8_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,512]
    conv10_left=cnn.conv_layer_init("conv10",conv9_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    conv10_1_left=cnn.conv_layer_init("conv10_1",conv10_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,512,1024]
    conv11_left=cnn.conv_layer_init("conv11",conv10_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,1024,1024]
    conv12_left=cnn.conv_layer_init("conv12",conv11_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    conv12_1_left=cnn.conv_layer_init("conv12_1",conv12_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    # shape=[3,3,1024,2048]
    # conv13_left=cnn.conv_layer_init("conv13",conv12_1_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    # shape=[3,3,2048,2048]
    # conv14_left=cnn.conv_layer_init("conv14",conv13_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    # conv14_1_left=cnn.conv_layer_init("conv14_1",conv14_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)

    conv5_right=cnn.conv_layer_reuse("conv5",conv4_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv6_right=cnn.conv_layer_reuse("conv6",conv5_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv6_1_right=cnn.conv_layer_reuse("conv6_1",conv6_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv7_right=cnn.conv_layer_reuse("conv7",conv6_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv8_right=cnn.conv_layer_reuse("conv8",conv7_right,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,training=training)
    conv8_1_right=cnn.conv_layer_reuse("conv8_1",conv8_right,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,training=training)

    conv9_right=cnn.conv_layer_reuse("conv9",conv8_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv10_right=cnn.conv_layer_reuse("conv10",conv9_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv10_1_right=cnn.conv_layer_reuse("conv10_1",conv10_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    conv11_right=cnn.conv_layer_reuse("conv11",conv10_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv12_right=cnn.conv_layer_reuse("conv12",conv11_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    conv12_1_right=cnn.conv_layer_reuse("conv12_1",conv12_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)

    # conv13_right=cnn.conv_layer_reuse("conv13",conv12_1_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    # conv14_right=cnn.conv_layer_reuse("conv14",conv13_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    # conv14_1_right=cnn.conv_layer_reuse("conv14_1",conv14_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)


    concat_features=tf.concat([conv12_1_left,conv12_1_right],3)#,conv5_right_strided_3,conv5_right_strided_7,conv5_right_strided_11],3)
    # shape=[1,1,160,160]
    # concat_features=cnn.conv_layer_init("concat_features",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,2048,2048]
    decod_conv1=cnn.conv_layer_init("decod_conv1",concat_features,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv2=cnn.conv_layer_init("decod_conv2",decod_conv1,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv2_2=cnn.conv_layer_init("decod_conv2_2",decod_conv2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,2048,1]
    final_score_1=cnn.conv_layer_init("final_score_1",decod_conv2_2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_1=tf.image.resize_images(final_score_1,[int(rows),int(cols)])


    deconv1=cnn.deconv_layer("deconv1",decod_conv2_2,1024,filter_size=3,stride=2,output_shape=[batch_size,int(rows/32),int(cols/32),1024])
    concat_features_2=tf.concat([deconv1,conv10_1_left,conv10_1_right],3)

    shape=[3,3,2048,1024]
    decod_conv3=cnn.conv_layer_init("decod_conv3",concat_features_2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,1024,1024]
    decod_conv4=cnn.conv_layer_init("decod_conv4",decod_conv3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv4_2=cnn.conv_layer_init("decod_conv4_2",decod_conv4,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,1024,1]
    final_score_2=cnn.conv_layer_init("final_score_2",decod_conv4_2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_2=tf.image.resize_images(final_score_2,[int(rows),int(cols)])


    deconv2=cnn.deconv_layer("deconv2",decod_conv4_2,512,filter_size=3,stride=2,output_shape=[batch_size,int(rows/16),int(cols/16),512])
    concat_features_3=tf.concat([deconv2,conv8_1_left,conv8_1_right],3)

    shape=[3,3,1024,512]
    decod_conv5=cnn.conv_layer_init("decod_conv5",concat_features_3,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,512]
    decod_conv6=cnn.conv_layer_init("decod_conv6",decod_conv5,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv6_1=cnn.conv_layer_init("decod_conv6_1",decod_conv6,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,512,1]
    final_score_3=cnn.conv_layer_init("final_score_3",decod_conv6_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_3=tf.image.resize_images(final_score_3,[int(rows),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    deconv3=cnn.deconv_layer("deconv3",decod_conv6_1,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),256])
    concat_features_4=tf.concat([deconv3,conv6_1_left,conv6_1_right],3)

    shape=[3,3,512,256]
    decod_conv7=cnn.conv_layer_init("decod_conv7",concat_features_4,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    decod_conv8=cnn.conv_layer_init("decod_conv8",decod_conv7,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv8_1=cnn.conv_layer_init("decod_conv8_1",decod_conv8,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,256,1]
    final_score_4=cnn.conv_layer_init("final_score_4",decod_conv8_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_4=tf.image.resize_images(final_score_4,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    deconv4=cnn.deconv_layer("deconv4",decod_conv8_1,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),128])
    concat_features_5=tf.concat([deconv4,conv4_left,conv4_right],3)

    shape=[3,3,256,128]
    decod_conv9=cnn.conv_layer_init("decod_conv9",concat_features_5,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    decod_conv10=cnn.conv_layer_init("decod_conv10",decod_conv9,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv10_1=cnn.conv_layer_init("decod_conv10_1",decod_conv10,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,128,1]
    final_score_5=cnn.conv_layer_init("final_score_5",decod_conv10_1,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_5=tf.image.resize_images(final_score_5,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    deconv5=cnn.deconv_layer("deconv5",decod_conv10,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),64])
    concat_features_6=tf.concat([deconv5,conv2_left,conv2_right],3)

    shape=[3,3,128,64]
    decod_conv11=cnn.conv_layer_init("decod_conv11",concat_features_6,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    decod_conv12=cnn.conv_layer_init("decod_conv12",decod_conv11,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv13=cnn.conv_layer_init("decod_conv13",decod_conv12,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,64,1]
    final_score_6=cnn.conv_layer_init("final_score_6",decod_conv13,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    final_score_6=tf.image.resize_images(final_score_6,[int(rows/1),int(cols/1)])#,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    deconv6=cnn.deconv_layer("deconv6",decod_conv13,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),64])
    concat_features_7=tf.concat([deconv6,x_left,x_right],3)

    shape=[3,3,70,32]
    decod_conv14=cnn.conv_layer_init("decod_conv14",concat_features_7,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    decod_conv15=cnn.conv_layer_init("decod_conv15",decod_conv14,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    decod_conv16=cnn.conv_layer_init("decod_conv16",decod_conv15,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[3,3,32,1]
    final_score=cnn.conv_layer_init("final_score_7",decod_conv16,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    # resized1=tf.image.resize_images(final_score_1,[int(rows/1),int(cols/1)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # resized2=tf.image.resize_images(final_score_2,[int(rows/1),int(cols/1)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # resized3=tf.image.resize_images(final_score_3,[int(rows/1),int(cols/1)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # resized4=tf.image.resize_images(final_score_4,[int(rows/1),int(cols/1)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # resized5=tf.image.resize_images(final_score_5,[int(rows/1),int(cols/1)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # concat_scores=tf.concat([resized1,resized2,resized3,resized4,resized5,final_score],3)
    #
    # shape=[3,3,6,32]
    # refinepyramid_4patch_alllosses_biggerfilter100001times.ckpt-100002000021=cnn.conv_layer_init("refine1",concat_scores,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    # shape=[3,3,32,32]
    # refine2=cnn.conv_layer_init("refine2",refine1,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    # refine3=cnn.conv_layer_init("refine3",refine2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)
    # shape=[3,3,32,1]
    # final_refine=cnn.conv_layer_init("final_refine",refine2,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)


    if training:
        loss_1=cnn.loss("loss_1", final_score_1,y,loss_type="regression")
        loss_2=cnn.loss("loss_2", final_score_2,y,loss_type="regression")
        loss_3=cnn.loss("loss_3", final_score_3,y,loss_type="regression")
        loss_4=cnn.loss("loss_4", final_score_4,y,loss_type="regression")
        loss_5=cnn.loss("loss_5", final_score_5,y,loss_type="regression")
        loss_6=cnn.loss("loss_6", final_score_6,y,loss_type="regression")
        loss_7=cnn.loss("loss_7", final_score,y,loss_type="regression")

        #loss_r=cnn.loss("loss_r", final_refine,y,loss_type="regression")
        loss=loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7
        tf.summary.scalar("loss", loss)

        return loss,final_score
    else:

        return final_score,final_score_1,final_score_2,final_score_3

def init_new_cnn_strided(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])


    shape=[5,5,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    print("conv2_left",conv2_left.get_shape())
    shape=[5,5,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=batch_norm,activation=batch_norm,padding=padding,w_init="xavier",training=training)

    #conv4_left=cnn.deconv_layer("deconv1_1",conv4_left,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),32])
    #
    #shape=[3,3,32,32]
    #conv5_left=cnn.conv_layer_init("conv4_1",conv4_left,shape,batch_norm=True,activation=True,padding=padding,w_init="xavier",training=training)
    #
    # conv5_left=cnn.deconv_layer("deconv1_2",conv5_left,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    #
    # conv5_left=cnn.conv_layer_init("conv4_2",conv5_left,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)



    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,stride=[1,1],padding=padding,training=training)
    #
    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=batch_norm,activation=batch_norm,stride=[1,1],padding=padding,training=training)

    #conv4_right=cnn.deconv_layer_reuse("deconv1_1",conv4_right,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),32])
    #conv5_right=cnn.conv_layer_reuse("conv4_1",conv4_right,batch_norm=True,activation=True,stride=[1,1],padding=padding,training=training)
    #
    # conv5_right=cnn.deconv_layer_reuse("deconv1_2",conv5_right,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])
    #
    # conv5_right=cnn.conv_layer_reuse("conv4_2",conv5_right,batch_norm=False,activation=False,stride=[1,1],padding=padding,training=training)


    # conv5_right_strided_3=cnn.conv_layer_reuse("conv4_1",conv5_right,batch_norm=False,activation=False,stride=[1,2],padding=padding,training=training)
    # left_c=int(conv5_right.get_shape()[2])
    # conv5_right_strided_3=tf.pad(conv5_right_strided_3, [[0,0],[0,0], [left_c-int(conv5_right_strided_3.get_shape()[2]),0],[0,0]], "CONSTANT")
    #
    # conv5_right_strided_7=cnn.conv_layer_reuse("conv4_1",conv5_right,batch_norm=False,activation=False,stride=[1,3],padding=padding,training=training)
    # conv5_right_strided_7=tf.pad(conv5_right_strided_7, [[0,0],[0,0], [left_c-int(conv5_right_strided_7.get_shape()[2]),0],[0,0]], "CONSTANT")
    #
    # conv5_right_strided_11=cnn.conv_layer_reuse("conv4_1",conv5_right,batch_norm=False,activation=False,stride=[1,5],padding=padding,training=training)
    # conv5_right_strided_11=tf.pad(conv5_right_strided_11, [[0,0],[0,0], [left_c-int(conv5_right_strided_11.get_shape()[2]),0],[0,0]], "CONSTANT")

    #concat_features=cnn.correlation_map(conv5_left, conv5_right, int(nclasses/2))



    concat_features=tf.concat([conv4_left,conv4_right],3)#,conv5_right_strided_3,conv5_right_strided_7,conv5_right_strided_11],3)
    # shape=[1,1,160,160]
    # concat_features=cnn.conv_layer_init("concat_features",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    print("concat_features",concat_features.get_shape())
    shape=[1,1,128,64]
    conv5=cnn.conv_layer_init("conv5",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv5_1=cnn.conv_layer_init("conv5_1",conv5,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    print("conv5_1",conv5_1.get_shape())

    shape=[3,3,64,128]
    conv6=cnn.conv_layer_init("conv6",conv5_1,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    conv6_1=cnn.conv_layer_init("conv6_1",conv6,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    print("conv6_1",conv6_1.get_shape())

    shape=[3,3,128,256]
    conv7=cnn.conv_layer_init("conv7",conv6_1,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    conv7_1=cnn.conv_layer_init("conv7_1",conv7,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)


    shape=[3,3,256,512]
    conv8=cnn.conv_layer_init("conv8",conv7_1,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,512,512]
    conv8_1=cnn.conv_layer_init("conv8_1",conv8,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    # shape=[3,3,512,1024]
    # conv9=cnn.conv_layer_init("conv9",conv8_1,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    # shape=[3,3,1024,1024]
    # conv9_1=cnn.conv_layer_init("conv9_1",conv9,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    # shape=[3,3,1024,1024]
    # conv9_2=cnn.conv_layer_init("conv9_2",conv9_1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    # shape=[3,3,512,1024]
    # conv8=cnn.conv_layer_init("conv8",conv7_2,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    # shape=[3,3,1024,1024]
    # conv8_1=cnn.conv_layer_init("conv8_1",conv8,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    # conv8_2=cnn.conv_layer_init("conv8_2",conv8_1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)


    deconv1=cnn.deconv_layer("deconv1",conv8_1,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/16),int(cols/16),256])#tf.image.resize_images(conv9_2,[int(rows/16),int(cols/16)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#cnn.deconv_layer("deconv1",conv7_2,512,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),512], padding='SAME',trainable=True,w_init="xavier")
    #deconv1 = tf.nn.relu(deconv1)

    shape=[3,3,256,256]
    deconv1_conv1=cnn.conv_layer_init("deconv1_conv1",deconv1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv_concat_1=tf.concat([deconv1_conv1,conv7_1],3)
    shape=[3,3,512,256]
    deconv1_conv2=cnn.conv_layer_init("deconv1_conv2",deconv_concat_1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,256,256]
    deconv1_conv3=cnn.conv_layer_init("deconv1_conv3",deconv1_conv2,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    print("deconv1_conv1",deconv1_conv1.get_shape())

    deconv2=cnn.deconv_layer("deconv2",deconv1_conv3,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/8),int(cols/8),128])#tf.image.resize_images(deconv1_conv3,[int(rows/8),int(cols/8)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#cnn.deconv_layer("deconv2",deconv1_conv2,256,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),256], padding='SAME',trainable=True,w_init="xavier")
    #deconv2 = tf.nn.relu(deconv2)

    #deconv_concat_2=tf.concat([deconv2,conv5_1],3)
    shape=[3,3,128,128]
    deconv2_conv1=cnn.conv_layer_init("decon2_conv1",deconv2,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv_concat_2=tf.concat([deconv2_conv1,conv6_1],3)

    shape=[3,3,256,128]
    deconv2_conv2=cnn.conv_layer_init("decon2_conv2",deconv_concat_2,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,128,128]
    deconv2_conv3=cnn.conv_layer_init("decon2_conv3",deconv2_conv2,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    print("deconv2_conv1",deconv2_conv1.get_shape())

    # #pdb.set_trace()
    deconv3=cnn.deconv_layer("deconv3",deconv2_conv3,64,filter_size=3,stride=2,output_shape=[batch_size,int(rows/4),int(cols/4),64])#tf.image.resize_images(deconv2_conv3,[int(rows/4),int(cols/4)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#cnn.deconv_layer("deconv3",deconv2_conv2,128,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),128], padding='SAME',trainable=True,w_init="xavier")
    # # #deconv3 = tf.nn.relu(deconv3)
    # #
    # # #concat_3=tf.concat([deconv3,conv5_1],3)
    shape=[3,3,64,64]
    deconv3_conv1=cnn.conv_layer_init("deconv3_conv1",deconv3,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv_concat_3=tf.concat([deconv3_conv1,conv5_1],3)
    shape=[3,3,128,64]
    deconv3_conv2=cnn.conv_layer_init("deconv3_conv2",deconv_concat_3,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    deconv3_conv3=cnn.conv_layer_init("deconv3_conv3",deconv3_conv2,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)


    deconv4=cnn.deconv_layer("deconv4",deconv3_conv3,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/2),int(cols/2),32])#tf.image.resize_images(deconv3_conv3,[int(rows/2),int(cols/2)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    shape=[3,3,32,32]
    deconv4_conv1=cnn.conv_layer_init("deconv4_conv1",deconv4,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    deconv4_conv1=tf.concat([deconv4_conv1,conv2_left],3)
    shape=[3,3,64,32]
    deconv4_conv2=cnn.conv_layer_init("deconv4_conv2",deconv4_conv1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    deconv4_conv3=cnn.conv_layer_init("deconv4_conv3",deconv4_conv2,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    #
    deconv5=cnn.deconv_layer("deconv5",deconv4_conv3,32,filter_size=3,stride=2,output_shape=[batch_size,int(rows/1),int(cols/1),32])#tf.image.resize_images(deconv3_conv3,[int(rows/2),int(cols/2)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    deconv5=tf.concat([deconv5,x_left],3)
    shape=[3,3,35,32]
    deconv5_conv1=cnn.conv_layer_init("deconv5_conv1",deconv5,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    deconv5_conv2=cnn.conv_layer_init("deconv5_conv2",deconv5_conv1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    shape=[1,1,32,1]
    final_score=cnn.conv_layer_init("final_score",deconv5_conv2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    #final_score=tf.image.resize_images(final_score,[int(rows/1),int(cols/1)],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #y=tf.squeeze(y,-1)
    # #refinement stage
    # deconv_ref=tf.concat([x_left,final_score],3)
    # shape=[3,3,4,32]
    # r_input=cnn.conv_layer_init("deconv_ref_conv1",deconv_ref,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    #
    # shape=[3,3,32,32]
    # r_conv1=cnn.conv_layer_init("r_conv1",r_input,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    # r_conv2=cnn.conv_layer_init("r_conv2",r_conv1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    # shape=[1,1,32,1]
    # r_final=cnn.conv_layer_init("r_final",r_conv2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)
    # # #

    if training:
        #valid_match=soft_argmin[:,:,right_w-left_h::]
        loss_1=cnn.loss("loss_1", final_score,y,loss_type="regression")
        #loss_2=cnn.loss("loss_2", r_final,y,loss_type="regression")

        loss=loss_1#+loss_2

        return loss,final_score
    else:

        return final_score

def init_new_cnn_3d(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True):
    input_dim=x_left.get_shape()
    batch_size=int(input_dim[0])
    rows=int(input_dim[1])
    cols=int(input_dim[2])

    shape=[5,5,3,32]
    conv1_left=cnn.conv_layer_init("conv1",x_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,32,32]
    conv2_left=cnn.conv_layer_init("conv2",conv1_left,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)

    print("conv2_left",conv2_left.get_shape())
    shape=[5,5,32,64]
    conv3_left=cnn.conv_layer_init("conv3",conv2_left,shape,batch_norm=batch_norm,stride=[2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,64,64]
    conv4_left=cnn.conv_layer_init("conv4",conv3_left,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)

    print("conv4_left",conv4_left.get_shape())


    conv1_right=cnn.conv_layer_reuse("conv1",x_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv2_right=cnn.conv_layer_reuse("conv2",conv1_right,batch_norm=batch_norm,padding=padding,training=training)

    conv3_right=cnn.conv_layer_reuse("conv3",conv2_right,batch_norm=batch_norm,stride=[2,2],padding=padding,training=training)
    conv4_right=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=False,activation=False,padding=padding,training=training)

    #conv4_right_strided_3=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=batch_norm,stride=[1,7],padding=padding,training=training)
    #left_c=int(conv3_right.get_shape()[2])
    #conv4_right_strided_3=tf.pad(conv4_right_strided_3, [[0,0],[0,0], [left_c-int(conv4_right_strided_3.get_shape()[2]),0],[0,0]], "CONSTANT")

    #conv4_right_strided_7=cnn.conv_layer_reuse("conv4",conv3_right,batch_norm=batch_norm,stride=[1,7],padding=padding,training=training)
    #conv4_right_strided_7=tf.pad(conv4_right_strided_7, [[0,0],[0,0], [left_c-int(conv4_right_strided_7.get_shape()[2]),0],[0,0]], "CONSTANT")

    #concat_features=tf.concat([conv4_left,conv4_right],3)#,conv4_right_strided_3,conv4_right_strided_7],3)
    cost_volume=[]
    conv4_col=int(conv4_right.get_shape()[2])
    conv4_right=tf.pad(conv4_right,[[0,0],[0,0], [int(nclasses/4),0],[0,0]], "CONSTANT")
    for d in range(int(nclasses/4)):
        #right=tf.pad(conv4_right[:,:,d::,:],[[0,0],[0,0], [d,0],[0,0]], "CONSTANT")
        right=tf.slice(conv4_right, begin = [0, 0, d, 0], size = [-1, -1, conv4_col, -1])
        cost_volume.append(tf.concat([conv4_left,right],3))
    concat_features=tf.stack(cost_volume,axis=1)


    print("concat_features",concat_features.get_shape())

    shape=[3,3,3,128,128]
    conv5=cnn.conv3d_layer_init("conv5",concat_features,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,3,128,128]
    conv5_1=cnn.conv3d_layer_init("conv5_1",conv5,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    print("conv5_1",conv5_1.get_shape())

    shape=[3,3,3,128,256]
    conv6=cnn.conv3d_layer_init("conv6",conv5_1,shape,batch_norm=batch_norm,stride=[2,2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,3,256,256]
    conv6_1=cnn.conv3d_layer_init("conv6_1",conv6,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    print("conv6_1",conv6_1.get_shape())

    shape=[3,3,3,256,512]
    conv7=cnn.conv3d_layer_init("conv7",conv6_1,shape,batch_norm=batch_norm,stride=[2,2,2],padding=padding,w_init="xavier",training=training)
    shape=[3,3,3,512,512]
    conv7_1=cnn.conv3d_layer_init("conv7_1",conv7,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    print("conv7_1",conv7_1.get_shape())

    deconv1=cnn.deconv3d_layer("deconv1",conv7_1,256,filter_size=3,stride=[2,2,2],output_shape=[batch_size,int(nclasses/8),int(rows/8),int(cols/8),256], padding='SAME',trainable=True,w_init="xavier")

    deconv_concat_1=tf.concat([deconv1,conv6_1],4)
    shape=[3,3,3,512,256]
    deconv1_conv1=cnn.conv3d_layer_init("deconv1_conv1",deconv_concat_1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,3,256,256]
    deconv1_conv2=cnn.conv3d_layer_init("deconv1_conv2",deconv1_conv1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    #print("deconv1_conv1",deconv1_conv1.get_shape())

    deconv2=cnn.deconv3d_layer("deconv2",deconv1_conv2,128,filter_size=3,stride=[2,2,2],output_shape=[batch_size,int(nclasses/4),int(rows/4),int(cols/4),128], padding='SAME',trainable=True,w_init="xavier")
    #deconv2 = tf.nn.relu(deconv2)

    deconv_concat_2=tf.concat([deconv2,conv5_1],4)
    shape=[3,3,3,256,128]
    deconv2_conv1=cnn.conv3d_layer_init("decon2_conv1",deconv_concat_2,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    shape=[3,3,3,128,128]
    deconv2_conv2=cnn.conv3d_layer_init("deconv2_conv2",deconv2_conv1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    #print("deconv2_conv1",deconv2_conv1.get_shape())

    #pdb.set_trace()
    deconv3=cnn.deconv3d_layer("deconv3",deconv2,1,filter_size=5,stride=[4,4,4],output_shape=[batch_size,int(nclasses/1),int(rows/1),int(cols/1),1],activation=False, padding='SAME',trainable=True,w_init="xavier")

    # #concat_3=tf.concat([deconv2_conv1,conv4_left],3)
    # shape=[3,3,3,64,64]
    # deconv3_conv1=cnn.conv3d_layer_init("deconv3_conv1",deconv3,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    # shape=[3,3,3,64,64]
    # deconv3_conv2=cnn.conv3d_layer_init("deconv3_conv2",deconv3_conv1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    #
    #
    # deconv4=cnn.deconv3d_layer("deconv4",deconv3,1,filter_size=3,stride=[2,2,2],output_shape=[batch_size,int(nclasses/1),int(rows/1),int(cols/1),1], padding='SAME',trainable=True,w_init="xavier")
    #deconv4 = tf.nn.relu(deconv4)
    #deconv_concat_4=tf.concat([deconv4,conv2_left],3)
    # shape=[3,3,3,32,32]
    # deconv4_conv1=cnn.conv3d_layer_init("deconv4_conv1",deconv4,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    # deconv4_conv2=cnn.conv3d_layer_init("deconv4_conv2",deconv4_conv1,shape,batch_norm=batch_norm,padding=padding,w_init="xavier",training=training)
    #print("deconv4_conv1",deconv4_conv1.get_shape())

    #shape=[1,1,1,32,1]
    #final_score=cnn.conv3d_layer_init("final_score",deconv4_conv2,shape,batch_norm=False,activation=False,padding=padding,w_init="xavier",training=training)


    final_score=tf.squeeze(deconv3,axis=-1)

    print("final_score",final_score.get_shape())

    index=np.array(range(0,nclasses),dtype=np.float32)
    index=index[np.newaxis,:,np.newaxis,np.newaxis]
    index=tf.constant(index)
    print("index",index.get_shape())

    soft_argmin=tf.multiply(final_score,index)
    print("soft_argmin",soft_argmin.get_shape())
    soft_argmin=tf.reduce_sum(soft_argmin,axis=1)
    print("soft_argmin",soft_argmin.get_shape())
    #refinement stage


    if training:
        #valid_match=soft_argmin[:,:,right_w-left_h::]
        loss=cnn.loss("loss_1", soft_argmin,y,loss_type="regression")

        return loss,soft_argmin
    else:

        return soft_argmin


def file_to_list(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def train_kitty(dataset,netPath="./bvlc_alexnet.npy",maxDisp=192,gpu='/gpu:1'):
    np.random.seed(seed=123)

    training_iter=100
    validation_iter=1000000
    validation_samples=1000
    #saving_iter=1000+1
    #max_epochs=10
    #iter_per_epoch=20000

    [training_list_left,training_list_right,training_list_noc_label,validation_list_left,validation_list_right,validation_list_noc_label]=dataset
    nclasses=maxDisp
    receptive_field=1#28#13#45#97
    batch_size=2
    model_name="pyramid_4_shallow_fullkitti2015"


    #------------------------------------kitti-----------------------------
    #left_images,right_images,disp_images=load_all_images(training_list_left,training_list_right,training_list_noc_label)
    #valid_pixels_train=get_valid_pixels(disp_images,receptive_field,maxDisp)
    #left_images_validation,right_images_validation,disp_images_validation=load_all_images(validation_list_left,validation_list_right,
    #validation_list_noc_label)

    # valid_pixels_val=get_valid_pixels(disp_images_validation,receptive_field,maxDisp)
    # np.save("left_images.npy",left_images)
    # np.save("right_images.npy",right_images)
    # np.save("disp_images.npy",disp_images)
    # np.save("left_images_validation.npy",left_images_validation)
    # np.save("right_images_validation.npy",right_images_validation)
    # np.save("disp_images_validation.npy",disp_images_validation)
    # np.save("valid_pixels_train.npy",valid_pixels_train)
    # np.save("valid_pixels_val.npy",valid_pixels_val)
    # exit(0)

    left_images=np.load("left_images.npy")
    right_images=np.load("right_images.npy")
    disp_images=np.load("disp_images.npy")
    valid_pixels_train=np.load("valid_pixels_train.npy")

    #left_images_validation=np.load("left_images_validation.npy")
    #right_images_validation=np.load("right_images_validation.npy")
    #disp_images_validation=np.load("disp_images_validation.npy")
    #valid_pixels_val=np.load("valid_pixels_val.npy")

    # left_images=np.append(left_images,left_images_validation)
    # right_images=np.append(right_images,right_images_validation)
    # disp_images=np.append(disp_images,disp_images_validation)
    # valid_pixels_train=np.append(valid_pixels_train,valid_pixels_val)

    #----------------------------------------- scene flow ----------------------------------------
    train_images=file_to_list("/home/pbrandao/datasets/scene_flow/train_images.txt")
    train_labels=file_to_list("/home/pbrandao/datasets/scene_flow/train_labels.txt")
    test_images=file_to_list("/home/pbrandao/datasets/scene_flow/test_images.txt")
    test_labels=file_to_list("/home/pbrandao/datasets/scene_flow/test_labels.txt")

    ft3d_dataset = ft3d_filenames("/home/pbrandao/datasets/scene_flow")



    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    training_log=""
    #with tf.device(gpu):
    #right_size=receptive_field+maxDisp#sample_size+maxDisp
    # x_left = tf.placeholder(tf.float32, (batch_size,)+(receptive_field,receptive_field,3))
    # x_right = tf.placeholder(tf.float32, (batch_size,)+(receptive_field,right_size,3))
    # y = tf.placeholder(tf.int32,  (batch_size,)+ (receptive_field,receptive_field))
    size=[384,1248]
    #size=[544,960]
    #size=[512,928]
    #size=[320,640]
    x_left = tf.placeholder(tf.float32, (batch_size,size[0],size[1],3))
    x_right = tf.placeholder(tf.float32, (batch_size,size[0],size[1],3))
    y = tf.placeholder(tf.float32,  (batch_size,size[0],size[1]))
    #ft3d_dataset["TEST"]=ft3d_dataset["TEST"][0::100]
    #training_mode = tf.placeholder_with_default(shape=(), input=True, name="training_mode")
    #input_pipeline_batch = input_pipeline(ft3d_dataset["TRAIN"], input_size=size,batch_size=batch_size,shuffle=True)
    # val_pipeline = input_pipeline(ft3d_dataset["TEST"], input_size=size,batch_size=batch_size,shuffle=False)
    # input_pipeline_batch=tf.cond(training_mode,
    #                       lambda: train_pipeline,
    #                       lambda: val_pipeline)
    #x_left, x_right, y=input_pipeline_batch
    #with tf.device('/gpu:2'):
    loss,match=init_shallow(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True)#init_new_cnn_strided(x_left,x_right,y,nclasses,batch_norm=True,padding="SAME",training=True)


    learning_start=1e-3
    learning_decay=0.1
    #log_file_name=model_name+"_batch_size_"+str(batch_size)+"_receptivefield_"+"_learningrate_"+str(learning_start)+"_learning_decay_"+str(learning_decay)+".txt"

    learning_rate =learning_start#tf.placeholder(tf.float32, shape=[])

    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth=True
    #config = tf.ConfigProto( allow_soft_placement=True,log_device_placement=True)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))  # defaults to saving all variables - in this case w and b
    #tf.histogram_summary("grad",grads_and_vars)
    lowest_error=1e10

    #with tf.variable_scope("finetune_kitti_") as scope:
    optimizer =cnn.optimizer(loss,learning_rate,optimizer_type="Adam")

    # negative_mining_index=[]
    # with open('negative_mining_full.txt', 'r') as file:
    #     for item in file:
    #         negative_mining_index.append(int(item))

    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"./pyramid_2_shallow_divided_negative_mining50001times.ckpt-2500100002")
        #optimizer =cnn.optimizer(loss,learning_rate,optimizer_type="Adam")


        #print(tf.global_variables_initializer())
        #for key in sorted(var_to_shape_map):
        #    print("tensor_name: ", key)
        #tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter(model_name+"_tensorboard", graph=tf.get_default_graph())
        #maxNoise=20

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        counter=0
        #file = open(log_file_name, 'w')
        for sav_times in [500001,50002,50003,5004]:#24000,8000,8001,8002]:
            for i in range(sav_times+1):
                #image_left,image_right,im_gt=load_full_images(left_images,right_images,disp_images,receptive_field,
                #maxDisp,batch_size,valid_pixels_train)
                image_left,image_right,im_gt=load_full_images(left_images,right_images,disp_images,maxDisp,batch_size,size)

                #start = time.time()
                #image_left,image_right,im_gt=load_scene_flow_images(train_images,train_labels,batch_size,size,negative_mining_index)


                #_,loss_val,res,im_gt=sess.run([optimizer,loss,match,y], feed_dict={training_mode: True})

                _,loss_val,res=sess.run([optimizer,loss,match], feed_dict={x_left: image_left,x_right:image_right, y: im_gt})

                #end = time.time()
                #print("time ",end - start)
                # plt.figure(1)
                # plt.imshow(res_l[0])
                # plt.colorbar()
                # plt.figure(2)
                # plt.imshow(res_r[0])
                # plt.colorbar()
                # plt.figure(3)
                # plt.imshow(res_gt[0])
                # plt.colorbar()
                # plt.show()
                if(i%training_iter==0):

                    # mean_error=pixel3error=0
                    # for a in range(100):
                    #         res,im_gt=sess.run([match,y], feed_dict={training_mode: False})
                    #         res=np.squeeze(res, axis=3)
                    #         mean_error+=np.mean(np.abs(res-im_gt))
                    #         pixel3error+=error(res,im_gt,3)
                    # if (mean_error<lowest_error):
                    #     saver.save(sess, "lowest_error",global_step=sav_times*i+1)
                    #     print(" new lowest error at iteration ",i, " error ",mean_error, " 3pixelerror ",pixel3error)
                    #     lowest_error=mean_error
                    res=np.squeeze(res, axis=3)
                    #res=np.argmax(res,3)
                    #error(res,im_gt,3)
                    error_it=np.abs(res-im_gt)
                    error_it[im_gt==0]=0
                    print(sav_times,i,loss_val,np.mean(error_it),error(res,im_gt,3))
                    #error_it[error_it>3]=255
                    #file.write("Training "+str(sav_times)+" "+str(i)+" "+str(loss_val)+"\n")
                    #    print("WROOOOOOOOOOOOOOOONG!")
                    error_it[error_it>3]=255
                    scipy.misc.imsave(model_name+".png",res[0,:,:])
                    scipy.misc.imsave(model_name+"gt.png",im_gt[0,:,:])
                    scipy.misc.imsave(model_name+"3pixelerror.png",error_it[0,:,:])
                    #scipy.misc.imsave(model_name+"error_3pixel.png",error_it[0,:,:])
                    #scipy.misc.imsave(model_name_"+gt_22loss_scenflow2.png",im_gt[1,:,:])
                    #scipy.misc.imsave("right_right.png",image_right[0,:,-np.argmax(im_gt,3)[0]-sample_size:-np.argmax(im_gt,3)[0],:])
                    #scipy.misc.imsave("right_predicted.png",image_right[0,:,-np.argmax(res,3)[0]-sample_size:-np.argmax(res,3)[0],:])
                if(loss_val==0):
                    #print(res,sav_times,i,np.argmax(im_gt,3)[0],np.argmax(res,3)[0])
                    #res=np.squeeze(res, axis=3)
                    #print(res.shape,np.max(np.max(res)),np.max(np.max(im_gt)))
                    print("Loss zero!!")
                    # scipy.misc.imsave("zeroloss"+".png",res[0])
                    # scipy.misc.imsave("zeroloss_gt.png",im_gt[0,:,:])
                    # print(np.max(np.max(res)),np.max(np.max(im_gt)))

                    #exit(0)
                # if(loss_val!=loss_val):
                #     #print(res)
                #     print("Loss not a number!!")
                #     exit(0)

                if (i%validation_iter==0) and (i>0):
                     total_val_loss=0.0
                     error_val=0.0
                     for n in range(validation_samples):
                         #image_left,image_right,im_gt=load_random_patch(left_images_validation,right_images_validation,disp_images_validation,
                         #receptive_field,maxDisp,batch_size,valid_pixels_val)
                         #image_left,image_right,im_gt=load_full_images(left_images_validation,right_images_validation,disp_images_validation,
                         #maxDisp,batch_size,size)
                         image_left,image_right,im_gt=load_scene_flow_images(test_images,test_labels,batch_size,size)

                         res=sess.run(match, feed_dict={x_left: image_left,x_right:image_right, y: im_gt})
                         total_val_loss+=loss_val
                         #error_val+=error(np.argmax(res,3),im_gt,3)
                         res=np.squeeze(res, axis=3)
                         error_val+=error(res,im_gt,3)
                     print("\n\n\nvalidation: epoch ",sav_times,i," ",total_val_loss/validation_samples," error: ",error_val/validation_samples,"\n\n\n")

                     #file.write("Testing epoch "+str(sav_times)+" "+str(i)+" "+str(total_val_loss/validation_samples)+" error "+str(error_val/validation_samples)+"\n")

                #
                #     #print(i,displacement,np.argmax(im_gt,3),tf.argmax(res,3).eval(),loss_val)
                #
                #     exit(0)
                counter+=1
            learning_start=learning_start*learning_decay
            saver.save(sess, model_name+str(sav_times)+"times.ckpt",global_step=sav_times*i+1)

    #file.close()
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    np.random.seed(seed=123)
    train_kitty(load_kitti_2015("/home/pbrandao/datasets/kitti/data_scene_flow_2015"))
    #train_kitty(load_kitti_2012("/home/patrick/datasets/kitti/data_stereo_flow_2012"))
