# -*- coding:utf-8 -*-
from SentimentLSTM import *
from SentimentSLSTM import *
from tag_lstm import *
from collections import OrderedDict
import cPickle
if __name__ == "__main__":
    dataset, Wemb, word_idx_map, vocab, Temb, tagger_idx_map, tvocab = cPickle.load(open("mr.pkl", "rb"))
    train, valid, test = dataset

    options = OrderedDict()
    # set options
    options['word_dim'] = 300
    options['mem_dim'] = 300
    options['y_dim'] = numpy.max(train[1]) + 1
    options["Wemb"] = Wemb
    options["Temb"] = Temb
    options["use_dropout"] = True

    # LSTM
    #senti_lstm = SentimentLSTM(options,model='lstm')
    max_epochs = 10
    batch_size = 25
    lrate = 0.0001

    # tagged-lstm
    results = []
    test_errs = []
    for i in range(5):
        senti_lstm = TagLSTMSentiment(options, "tagger-lstm")
        train_err, valid_err, test_err = senti_lstm.train(dataset, max_epochs, batch_size, lrate)
        results.append([train_err, valid_err, test_err])
    for r in results:
        print r
    print "=" * 20
    test_errs = numpy.array(results)[:, 2]
    print "Accuary:%.3f(%.4f)" % ((1 - test_errs.mean(), test_errs.std()))


