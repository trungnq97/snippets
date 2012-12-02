#!/usr/bin/env python
# Dec 2012
__author__ = "Trung Nguyen, trungnq97@gmail.com"


if __name__ == "__main__":

    from pybrain.structure import RecurrentNetwork
    from pybrain.structure import LinearLayer
    from pybrain.structure import SigmoidLayer
    from pybrain.structure import FullConnection

    net = RecurrentNetwork()

    net.addInputModule(LinearLayer(2, "in"))
    net.addModule(SigmoidLayer(3, "hidden"))
    net.addOutputModule(LinearLayer(1, "out"))

    net.addConnection(FullConnection(net["in"], net["hidden"], "c1"))
    net.addConnection(FullConnection(net["hidden"], net["out"], "c2"))
    net.addRecurrentConnection(FullConnection(net["hidden"], net["hidden"],
        "c3-recurrent"))

    net.sortModules()

    print net

    for i in xrange(5):
        print net.activate([2, 2])

    print "reset"
    net.reset()

    for i in xrange(5):
        print net.activate([2, 2])
