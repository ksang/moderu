import argparse
import os
import torch

from torchvision import datasets, transforms
from model import LeNet5

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='LeNet-5 model parameter plotting')
parser.add_argument('-m', '--model', default='', type=str, metavar='PATH',
                    required=True,
                    help='path to saved model or checkpoint(default: none)')
parser.add_argument('-c', '--num-cols', default=6, type=int,
                    metavar='N', help='number of kernel plot for each column')
parser.add_argument('--data', metavar='DIR', default='data',
                    help='path to dataset')
parser.add_argument('-n', '--num-samples', type=int, default=3,
                    metavar='N', help='number of samples to plot feature maps')
parser.add_argument('-o', '--output', type=str, default='',
                    metavar='N', help='save plot images to directory')

def plot_conv_kernels(tensor, title, args):
    num_filters = tensor.shape[0] * tensor.shape[1]
    q, r = divmod(num_filters, args.num_cols)
    num_rows = q + int(bool(r))
    num_cols = args.num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    kernel = tensor.shape[2]
    plt.title(title)
    i = 0
    for t in tensor:
        for filter in t:
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            pilTrans = transforms.ToPILImage()
            pilImg = pilTrans(filter.reshape(kernel,kernel,1))
            ax1.imshow(pilImg, interpolation='none')
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            i+=1

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        plt.savefig(os.path.join(args.output, '{}.png'.format(title)))


def plot_feature_map(data, feature, title, args):
    num_feature_maps = feature.shape[0] * feature.shape[1] + 1
    q, r = divmod(num_feature_maps, args.num_cols)
    num_rows = q + int(bool(r))
    num_cols = args.num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    plt.title(title)
    width = feature.shape[2]
    height = feature.shape[3]

    pilTrans = transforms.ToPILImage()
    ax1 = fig.add_subplot(num_rows,num_cols,1)
    pilImg = pilTrans(data.reshape(1, data.shape[2],data.shape[3]))
    ax1.imshow(pilImg, interpolation='none')
    ax1.axis('off')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    i = 0
    for t in feature:
        for feature_map in t:
            ax2 = fig.add_subplot(num_rows,num_cols,i+2)
            pilImg = pilTrans(feature_map.reshape(1, width,height))
            ax2.imshow(pilImg, interpolation='none')
            ax2.axis('off')
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            i+=1

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        plt.savefig(os.path.join(args.output, '{}.png'.format(title)))

def normalize(filter):
    f_max, f_min = filter.max() , filter.min()
    return (filter - f_min) / (f_max - f_min) * 255

def main():
    args = parser.parse_args()
    model = torch.nn.DataParallel(LeNet5())
    if os.path.isfile(args.model):
        print("=> loading model '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model '{}'"
              .format(args.model))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    sample_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=False,
                        transform=transforms.Compose([
                           transforms.ToTensor()
                        ])),
        batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True)

    # plot convolotion layer kernels
    filter1 = model.module.conv1.weight.data.numpy()
    filter2 = model.module.conv2.weight.data.numpy()
    plot_conv_kernels(normalize(filter1), 'conv1_filters', args)
    plot_conv_kernels(normalize(filter2), 'conv2_filters', args)
    # plot feature maps
    for idx, (data, _) in enumerate(sample_loader):
        if idx == args.num_samples:
            break
        feature1 = model.module.feature_conv1(data)
        feature2 = model.module.feature_conv2(data)
        plot_feature_map(data, feature1,
            'sample_#{}_conv1_feature_maps'.format(idx+1), args)
        plot_feature_map(data, feature2,
            'sample_#{}_conv2_feature_maps'.format(idx+1), args)

    if not args.output:
        plt.show()

if __name__ == '__main__':
    main()
