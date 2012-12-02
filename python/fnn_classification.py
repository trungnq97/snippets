#!/usr/bin/env python
# Dec 2012
__author__ = "Trung Nguyen, trungnq97@gmail"


from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from pybrain.datasets import ClassificationDataSet

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal


if __name__ == "__main__":

    means = [(-1, 0), (2, 4), (3, 1)]
    cov = [diag([1, 1]), diag([0.5, 1.2]), diag([1.5, 0.7])]
    ds = ClassificationDataSet(2, 1, nb_classes=3)

    for n in xrange(400):
        for cl in xrange(3):
            input = multivariate_normal(means[cl], cov[cl])
            ds.addSample(input, [cl])

    # split dataset to train data & test data
    test_data, train_data = ds.splitWithProportion(0.25)

    print "train data", len(train_data)
    print "test data", len(test_data)

    # encode 1-k for target
    train_data._convertToOneOfMany()
    test_data._convertToOneOfMany()

    # build ffn
    net = buildNetwork(train_data.indim, 5, 4, train_data.outdim,
        outclass=SoftmaxLayer)

    # backprop trainer
    trainer = BackpropTrainer(net, train_data, momentum=0.1,
        weightdecay=0.01)

    # train
    for i in xrange(100):
        trainer.train()
