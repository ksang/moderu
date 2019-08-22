# moderu
A collection of deep learning model implementations


### LeNet-5

LeNet-5 distributed data parallelism implementation on MNIST data set.

Training on CPU or single GPU:

    python3 lenet5/train.py

Training on local node multi-GPU and data parallel:

    python3 lenet5/train.py --multiprocessing-distributed --rank 0 --world-size 1 --dist-url file:///tmp/sharedfile
