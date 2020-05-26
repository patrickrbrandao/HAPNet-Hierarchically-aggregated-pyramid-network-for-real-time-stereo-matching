# HAPNet-Hierarchically-aggregated-pyramid-network-for-real-time-stereo-matching
HAPNet is a deep learning architecture which avoids the explicit construction of a cost volume of similarity which is one of the most computationally costly blocks of stereo algorithms. This makes training our network significantly more efficient and avoids the needs for large memory allocation. Our method performs well, especially around regions compromising multiple discontinuities around surgical instrumentation or around complex small structures and instruments. The method compares well to the state-of-the-art techniques while taking a different methodological angle to computational stereo problem in surgical video.

## Pretrained weights
[KITTI 2015](https://drive.google.com/drive/folders/1a6v7R7kFdsbhQP6y0EBUVCJTn0oSPHy5?usp=sharing)

## Inference

Setup a conda environment using requirements.txt and activate it.
  
``` conda create -n hapnet --file ./requirements.txt```

Download the pretrained weights using the provided link.

Run the folowing:
  
```
python inference.py --left path_to_left_img \
                    --right path_to_right_img \
                    --model path_to_pretrained_weight_file \
                    --out path_to_store_resulting_disparity_img
```

