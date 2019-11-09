#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys


##########################################################

class Network(torch.nn.Module):
    def __init__(self, arguments_strModel):
        super(Network, self).__init__()

        self.moduleVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.moduleScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.moduleCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
    # end

    def forward(self, tensorInput):
        tensorBlue = (tensorInput[:, 0:1, :, :] * 255.0) - 104.00698793
        tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
        tensorRed = (tensorInput[:, 2:3, :, :] * 255.0) - 122.67891434

        tensorInput = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)

        tensorVggOne = self.moduleVggOne(tensorInput)
        tensorVggTwo = self.moduleVggTwo(tensorVggOne)
        tensorVggThr = self.moduleVggThr(tensorVggTwo)
        tensorVggFou = self.moduleVggFou(tensorVggThr)
        tensorVggFiv = self.moduleVggFiv(tensorVggFou)

        tensorScoreOne = self.moduleScoreOne(tensorVggOne)
        tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
        tensorScoreThr = self.moduleScoreThr(tensorVggThr)
        tensorScoreFou = self.moduleScoreFou(tensorVggFou)
        tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)

        tensorScoreOne = torch.nn.functional.interpolate(input=tensorScoreOne, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreTwo = torch.nn.functional.interpolate(input=tensorScoreTwo, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreThr = torch.nn.functional.interpolate(input=tensorScoreThr, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFou = torch.nn.functional.interpolate(input=tensorScoreFou, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
        tensorScoreFiv = torch.nn.functional.interpolate(input=tensorScoreFiv, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)

        return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ], 1))
    # end


##########################################################

def estimate(tensorInput, arguments_strModel):

    #   Establish network
    moduleNetwork = Network(arguments_strModel).cuda().eval()

    intWidth = tensorInput.size(2)
    intHeight = tensorInput.size(1)

    return moduleNetwork(tensorInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
# end

##########################################################

def edge_aligned(arguments_strModel, arguments_strIn, arguments_strOut, arguments_strSide, arguments_invert, arguments_binarize=False):

    assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    #   Initial image from .open has dims   H x W x C
    #   After transpose, dims are           C x H x W
    if arguments_strSide == "A":
        start=0
        end=256
    elif arguments_strSide == "B":
        start=256
        end=512

    image = numpy.array(PIL.Image.open(arguments_strIn))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)

    tensorInput = torch.FloatTensor(image[:, :, start:end])

    tensorOutput = estimate(tensorInput, arguments_strModel)

    if arguments_invert:
        if arguments_binarize:
            #   ^1 is bitwise xor
            image[:, :, start:end] = (tensorOutput.clamp(0.0, 1.0).round().int()^1).numpy()
            #image[:, :, start:end] = (tensorOutput.clamp(0.0, 1.0).ceil().int()^1).numpy()
        else:
            image[:, :, start:end] = (1 - tensorOutput.clamp(0.0, 1.0)).numpy()
    else:
        if arguments_binarize:
            image[:, :, start:end] = tensorOutput.clamp(0.0, 1.0).round().numpy()
            image[:, :, start:end] = tensorOutput.clamp(0.0, 1.0).ceil().numpy()
        else:
            image[:, :, start:end] = tensorOutput.clamp(0.0, 1.0).numpy()

    PIL.Image.fromarray((image.transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)
    # end edge_aligned
##########################################################

##########################################################

def edge_unaligned(arguments_strModel, arguments_strIn, arguments_strOut, arguments_invert, arguments_binarize=False):

    assert(int(str('').join(torch.__version__.split('.')[0:3])) >= 41) # requires at least pytorch version 0.4.1
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

    image = numpy.array(PIL.Image.open(arguments_strIn))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)

    tensorInput = torch.FloatTensor(image)

    tensorOutput = estimate(tensorInput, arguments_strModel)


    if arguments_invert:
        if arguments_binarize:
            #   ^1 is bitwise xor
            image = (tensorOutput.clamp(0.0, 1.0).round().int()^1).numpy()
            #image = (tensorOutput.clamp(0.0, 1.0).ceil().int()^1).numpy()
        else:
            image = (1 - tensorOutput.clamp(0.0, 1.0)).numpy()
    else:
        if arguments_binarize:
            image = tensorOutput.clamp(0.0, 1.0).round().numpy()
            #image = tensorOutput.clamp(0.0, 1.0).ceil().numpy()
        else:
            image = tensorOutput.clamp(0.0, 1.0).numpy()

    PIL.Image.fromarray((image.transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)
    # end edge_unaligned
##########################################################

if __name__ == '__main__':
    arguments_strModel = 'bsds500'
    arguments_strIn = './images/sample.png'
    arguments_strOut = './out.png'
    arguments_strSide = 'A'

    for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
        if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
        if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
        if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
        if strOption == '--side' and strArgument != '': arguments_strSide = strArgument # path to where the output should be stored

    edge(arguments_strModel, arguments_strIn, arguments_strOut, arguments_strSide)
# end main
