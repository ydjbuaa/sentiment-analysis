# -*- coding:utf-8 -*-
import theano
import numpy
import theano.tensor as tensor
w_values = numpy.random.randn(100,125)
w = theano.shared(value=w_values, name="w")
print w.get_value()
print w.name