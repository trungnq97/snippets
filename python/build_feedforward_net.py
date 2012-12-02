#!/usr/bin/env python
# Dec 2012
__author__ = "Trung Nguyen, trungnq97@gmail"

if __name__ == "__main__":

    from pybrain.structure import FeedForwardNetwork
    from pybrain.structure import LinearLayer
    from pybrain.structure import SigmoidLayer
    from pybrain.structure import FullConnection

    net = FeedForwardNetwork()

    # create layers
    in_layer = LinearLayer(2, "input-layer")        # 2 linear neurons
    hidden_layer = SigmoidLayer(3, "hidden-layer")  # 3 sigmoid neurons
    out_layer = LinearLayer(1, "output-layer")      # 1 linear neuron

    # add layers to network
    net.addInputModule(in_layer)
    net.addModule(hidden_layer)
    net.addOutputModule(out_layer)

    # connect layers using FullConnection
    in_hidden = FullConnection(in_layer, hidden_layer)
    hidden_out = FullConnection(hidden_layer, out_layer)

    # add connections to network
    net.addConnection(in_hidden)
    net.addConnection(hidden_out)

    # arrange things to make network usable
    net.sortModules()

    for m in net.modulesSorted:
        print m, "num of neurons %d" % m.dim

    print "all weights %d\n" % net.paramdim, net.params
    print "in_hidden connection weights\n", in_hidden.params
    print "hidden_out connection weights\n", hidden_out.params
