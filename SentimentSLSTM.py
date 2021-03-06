# -*- coding:utf-8 -*-
from models import *
from theano import config
import theano.tensor as tensor
import time
import numpy
# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def  prepare_data(seqs, labels, taggers, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        new_taggers = []
        for l, s, y, t in zip(lengths, seqs, labels, taggers):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
                new_taggers.append(t)

        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs
        taggers = new_taggers

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    xc = numpy.zeros((maxlen, n_samples)).astype('int64')
    xs = numpy.zeros((maxlen, n_samples)).astype('int64')
    xc_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    for idx, s in enumerate(seqs):
        xc[:lengths[idx], idx] = s
        xs[:lengths[idx], idx] = taggers[idx]
        xc_mask[:lengths[idx], idx] = 1.

    y = numpy.asarray(labels, dtype="int32")
    return xc, xc_mask, xs, xc_mask, y


def sepate_data(seqs, labels, taggers):
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)
    xc = numpy.zeros((maxlen, n_samples)).astype('int64')
    xs = numpy.zeros((maxlen, n_samples)).astype('int64')
    xc_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    xs_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    list = [17, 27, 41, 26, 35, 2, 18, 10, 44, 43]
    flags = set(list)

    for i in range(len(seqs)):
        s = seqs[i]
        for j in range(len(s)):
            w = s[j]
            t = taggers[i][j]
            if t in flags:
                xc[j,i]= w
                xc_mask[j, i] = 1.0
            else:
                xs[j,i] = w
                xs_mask[j, i] = 1.0

    return xc, xc_mask, xs, xc_mask, labels

def adadelta(lr, params, grads, xc, xs, xc_mask, xs_mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    pramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % p.name)
                    for p in params]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % p.name)
                   for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % p.name)
                      for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([xc, xs, xc_mask, xs_mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(params, updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update
class SentimentSLSTM(object):
    """
    use structural LSTM for sentiment analysis
    """
    def __init__(self, options, model):
        """
        build model

        """
        self.word_dim = options['word_dim']
        self.mem_dim = options['mem_dim']
        self.y_dim = options['y_dim']
        self.params = []
        self.model = model


        print ("Building Model:", model)

        # variables
        xc = tensor.matrix('xc', dtype='int64')
        xs = tensor.matrix("xs", dtype='int64')
        xc_mask = tensor.matrix('xc_mask', dtype=config.floatX)
        xs_mask = tensor.matrix("xs_mask", dtype=config.floatX)
        y = tensor.vector('y', dtype='int64')

        n_timesteps = xc.shape[0]
        n_samples = xc.shape[1]

        # add word embeddings to params list
        Wembs = theano.shared(options['Wemb'], "Wemb")
        self.params.append(Wembs)

        # get embeddings
        xc_emb = Wembs[xc.flatten()].reshape([n_timesteps, n_samples, self.word_dim])

        if model == "tagged_slstm":
            Tembs = theano.shared(options["Temb"], "Temb")
            self.params.append(Tembs)

            xs_emb = Tembs[xs.flatten()].reshape([n_timesteps, n_samples, self.word_dim])
        else:
            xs_emb = Wembs[xs.flatten()].reshape([n_timesteps, n_samples, self.word_dim])

        # init LSTM layer
        self.lstm_layer = SLSTM(input_dim=self.word_dim, hidden_dim=self.mem_dim, prefix=model)

        # add lstm params
        for param in self.lstm_layer.params:
            self.params.append(param)

        # get output hidden state from LSTM
        hc_state, hs_state = self.lstm_layer.layer_output(xc_emb, xs_emb, xc_mask, xs_mask)

        hc_state = (hc_state * xc_mask[:, :, None]).sum(axis=0)
        #if xc_mask.sum(axis=0)[:, None] > 0 :
        hc_state = hc_state / xc_mask.sum(axis=0)[:, None]

        #hs_state = (hs_state * xs_mask[:, :, None]).sum(axis=0)
        #if xs_mask.sum(axis=0)[:, None] > 0:
        #    hs_state = hs_state / xs_mask.sum(axis=0)[:, None]

        # Used for dropout.
        self.use_noise = theano.shared(numpy_floatX(0.))
        if options['use_dropout']:
            self.hidden_dropout_layer = DropoutLayer(state_before=hc_state, use_noise=self.use_noise)
            hc_state = self.hidden_dropout_layer.drop_out

        # init logistic regression layer
        self.logic_regression = LogisticRegression(input=hc_state,
                                                   input_size=self.mem_dim,
                                                   output_size=self.y_dim,
                                                   prefix="lr")

        """
        self.logic_regression = LogisticRegression2(x1=hc_state,
                                                    x2 = hs_state,
                                                   input_size=self.mem_dim,
                                                   output_size=self.y_dim,
                                                   prefix="lr")
        """
        # add logicstic regression params
        for param in self.logic_regression.params:
            self.params.append(param)

        pred = self.logic_regression.p_y_given_x

        f_pred_prob = theano.function([xc, xs, xc_mask, xs_mask], pred, name='f_pred_prob')
        self.f_pred = theano.function([xc, xs, xc_mask, xs_mask], pred.argmax(axis=1), name='f_pred')

        log_likelihood_cost = self.logic_regression.negative_log_likelihood(y)

        l2_sqr = self.lstm_layer.l2_sqr() + self.logic_regression.l2_sqr()
        cost = log_likelihood_cost # + 0.5 * 0.0001 * l2_sqr

        print('Optimization')
        f_cost = theano.function([xc, xs, xc_mask, xs_mask, y], cost, name='f_cost')

        grads = tensor.grad(cost, self.params)
        f_grad = theano.function([xc, xs, xc_mask, xs_mask, y], grads, name='f_grad')

        lr = tensor.scalar(name='lr')

        self.f_grad_shared, self.f_update = adadelta(lr, self.params, grads, xc, xs, xc_mask, xs_mask, y, cost)

    def pred_error(self, prepare_data, data, iterator, verbose=False):
        """
        Just compute the error
        f_pred: Theano fct computing the prediction
        prepare_data: usual prepare_data for that dataset.
        """
        valid_err = 0
        for _, valid_index in iterator:
            if self.model == "tagged_slstm":
                xc, xc_mask, xs, xs_mask, y = prepare_data([data[0][t] for t in valid_index],
                                          numpy.array(data[1])[valid_index],
                                          [data[2][t] for t in valid_index],
                                          maxlen=None)
            else:
                xc, xc_mask, xs, xs_mask, y = sepate_data([data[0][t] for t in valid_index],
                                                           numpy.array(data[1])[valid_index],
                                                           [data[2][t] for t in valid_index],
                                                           )
            preds = self.f_pred(xc, xs, xc_mask, xs_mask)
            targets = numpy.array(data[1])[valid_index]
            valid_err += (preds == targets).sum()
        valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

        return valid_err

    def train(self, dataset, max_epochs, batch_size, lrate, dispFreq=10, patience=5, saveto=None):

        train, valid, test = dataset
        kf_valid = get_minibatches_idx(len(valid[0]), batch_size)
        kf_test = get_minibatches_idx(len(test[0]), batch_size)

        print("%d train examples" % len(train[0]))
        print("%d valid examples" % len(valid[0]))
        print("%d test examples" % len(test[0]))

        history_errs = []
        best_p = None

        final_valid_err = 1.0
        final_test_err = 1.0
        final_train_err = 1.0

        bad_count = 0

        validFreq = len(train[0]) // batch_size

        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.time()
        try:
            for eidx in range(max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    self.use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    y = [train[1][t] for t in train_index]
                    x = [train[0][t] for t in train_index]
                    s = [train[2][t] for t in train_index]
                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    if self.model == "tagged_slstm":
                        xc, xc_mask, xs, xs_mask, y = prepare_data(x, y, s)
                    else:
                        xc, xc_mask, xs, xs_mask, y = sepate_data(x, y, s)

                    n_samples += xc.shape[1]
                    n_timesteps = xc.shape[0]
                    n_samples = xc.shape[1]

                    cost = self.f_grad_shared(xc, xs, xc_mask, xs_mask, y)
                    self.f_update(lrate)

                    if numpy.isnan(cost) or numpy.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.

                    if numpy.mod(uidx, dispFreq) == 0:
                        print('Epoch:%d, Update:%d, Cost:%f ' % (eidx, uidx, cost))

                    if numpy.mod(uidx, validFreq) == 0:
                        self.use_noise.set_value(0.)
                        train_err = self.pred_error(prepare_data, train, kf)
                        valid_err = self.pred_error(prepare_data, valid,
                                                    kf_valid)
                        test_err = self.pred_error(prepare_data, test, kf_test)

                        history_errs.append([valid_err, test_err])

                        if (best_p is None or
                                    valid_err <= numpy.array(history_errs)[:,
                                                 0].min()):
                            # print ("get best temp params, Epoch:%d" % eidx)
                            best_p = self.params
                            # update best result as final result
                            final_train_err = train_err
                            final_valid_err = valid_err
                            final_test_err = test_err

                            bad_counter = 0

                        print(('Train ', train_err, 'Valid ', valid_err,
                               'Test ', test_err))

                        if (len(history_errs) > patience and
                                    valid_err >= numpy.array(history_errs)[:-patience,
                                                 0].min()):
                            bad_count += 1
                            if bad_count > patience:
                                print('Early Stop!')
                                estop = True
                                break

                print('Seen %d samples' % n_samples)

                if estop:
                    break

        except KeyboardInterrupt:
            print("Training interupted")

        end_time = time.time()
        print('Train ', final_train_err, 'Valid ', final_valid_err, 'Test ', final_test_err)

        print('The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
        return final_test_err, final_valid_err, final_test_err