#!/usr/bin/env python
# Dec 2012
__author__ = "Trung Nguyen, trungnq97@gmail.com"

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork


def output(input):
    """return 1 or 0 based on output (param: input) from neural net
    """
    from numpy import argmax
    return argmax(input)


if __name__ == "__main__":

    # use SoftmaxLayer to have output encoding as
    # [1 0] corresponding to 0
    # [0 1] corresponding to 1
    # it is considered as classification problem with class 0 and class 1
    # convert output of net to real result using function output above
    net = buildNetwork(2, 5, 2, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
    ds = SupervisedDataSet(2, 2)

    # might need more data to train
    # but in this example, ds with 4 samples is pretty enough
    for i in xrange(1):
        ds.addSample((0.0, 0.0), (1.0, 0.0))
        ds.addSample((0.0, 1.0), (0.0, 1.0))
        ds.addSample((1.0, 0.0), (0.0, 1.0))
        ds.addSample((1.0, 1.0), (1.0, 0.0))

    print len(ds)

    trainer = BackpropTrainer(net, ds, learningrate=0.01, momentum=0.99)

    # train 100 epoches
    for i in xrange(100):
        print trainer.train()

    print "test on data", trainer.testOnData()

    # test
    print output(net.activate((0.0, 0.0)))  # 1 0 --> 0
    print output(net.activate((1.0, 0.0)))  # 0 1 --> 1
    print output(net.activate((0.0, 1.0)))  # 0 1 --> 1
    print output(net.activate((1.0, 1.0)))  # 1 0 --> 0
