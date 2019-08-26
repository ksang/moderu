# moderu
A collection of deep learning model implementations


### LeNet-5

LeNet-5 distributed data parallelism implementation on MNIST data set.

Training on CPU or single GPU:

    python3 lenet5/train.py

Training on local node multi-GPU and data parallel:

    python3 lenet5/train.py --multiprocessing-distributed       \
                            --rank 0 --world-size 1             \
                            --dist-url file:///tmp/sharedfile

Load model and plot convolution filters and feature maps for some samples:

    python3 lenet5/plot.py -m checkpoints/lenet5.tar            \
                           -c 6 -o output

This is how feature map looks like, the upper-left subplot is input data

![lenet5_conv1_feature_maps](/imgs/lenet5_conv1_feature_maps.png "conv1")

![lenet5_conv1_feature_maps](/imgs/lenet5_conv2_feature_maps.png "conv2")
