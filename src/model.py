import numpy as np
import math


LF = lambda x: 1 / (1 + np.exp(-x))
dLF = lambda x: LF(x) * (1 - LF(x))


SF = lambda x : np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
diSF = lambda x: SF(x) * (1 - SF(x))

loss = lambda t, o : - ( t * np.log(o))
dloss = lambda t, o : - ((t / o) + (1 - t) / (1 - o))

loss_square = lambda t, o : np.square( t - o )
dloss_square = lambda t,o : -2 * t + 2 * o

tanh = lambda x : (np.exp(x) - np.exp(-x)) / (np.exp(x) - np.exp(-x))
dtabh = lambda x : 1 - (tanh(x) * tanh(x))

act_equal = lambda x : x
deq = lambda x : 1
add1_x = lambda x: np.append(x, [[1]] * len(x) , axis = 1)

class Neuron():
    def __init__(self, count_x, count_y):
        self.weights = np.random.uniform(low = -1 * np.sqrt(6.0 / (count_x + count_y)), high = 1 * np.sqrt(6.0 / (count_x + count_y)),  size = (count_x, count_y))
        self.inX = np.zeros(count_x,)
        self.netX = np.zeros(count_x,)

class Layer():
    def __init__(self, last_layer = None,\
                 next_layer = None,\
                 is_out = False,\
                 is_in = False,\
                 is_hidden = False,\
                 neurons = None,\
                 activation = LF,\
                 loss = loss,\
                 dloss = dloss,\
                 use_CN = False):
        self.last_m = 0
        self.last_v = 0
        self.adam_learned = 0
        self.last_momentum = 0

        self.use_CN = use_CN

        self.loss = loss
        self.dloss = dloss
        self.neurons = neurons
        self.activation = activation
        self.inputs = []
        self.outputs = []
        self.delta = []
        self.last_layer = last_layer
        self.next_layer = next_layer
        self.is_out = is_out
        self.is_in = is_in
        self.is_hidden = is_hidden
        self.last_offset = None
        self.batch_offset = 0

    def forward(self, input):
        if(self.is_in):
            self.input = input
            self.neurons.inX = input
        self.neurons.netX = np.asarray(self.input).dot(self.neurons.weights)
        self.outputs = self.activation(x = self.neurons.netX)
        if(self.next_layer is not None):
            self.next_layer.input = self.outputs
        return self.outputs
        

    def backprop(self, target, lr, dloss, dact, dout_act):
        if(self.is_out):
            if(self.use_CN):
                weights_shape = self.neurons.weights.shape
                #dErW = dErNet * dNetrW
                #in Softmax with CN, use dErNet directly can reduece the accuracy loss.
                dErNet = self.outputs - target
                self.delta = dErNet * weights_shape[0]
                dNetrW = self.input

                dErW = self.delta * np.swapaxes( [ dNetrW ] * weights_shape[1], axis1 = 1, axis2 = 0)
                self.offset = dErW
            else:
                weights_shape = self.neurons.weights.shape

                dErO = dloss(t = target, o = self.outputs)
                dOrNet = dout_act(x = self.neurons.netX)


                dNetrW = self.input

                #dErW = dErO * dOrNet * dNetrW
                self.delta = [dErO * dOrNet] * weights_shape[0]
                dErW = self.delta * np.swapaxes( [ dNetrW ] * weights_shape[1], axis1 = 1, axis2 = 0)
                self.offset = dErW
        else:
            weights_shape = self.neurons.weights.shape

            dOrNet = dact(x = self.neurons.netX)

            dNetrW = self.input

            dErO = np.sum(self.next_layer.delta * self.next_layer.neurons.weights, axis = 1)

            self.delta = [dErO * dOrNet] * weights_shape[0]

            dErW = self.delta * np.swapaxes([ dNetrW ] * weights_shape[1], axis1 = 1, axis2 = 0)

            self.offset =  dErW


class model():
    def __init__(self, depth, inputsize, classes, lr, max_batch, target_acc,  width, mloss = loss, mdloss = dloss, act = LF, dact = dLF, out = SF, dout = diSF, optimizer = None, batch = True, batch_size = None, CN = False, screen = None):
        self.depth = depth
        self.inputsize = inputsize
        self.classes = classes
        self.lr = lr
        self.max_batch = max_batch
        self.target_acc = target_acc
        self.outs = []
        self.final_layer = None
        self.all_loss_last = [0]
        self.layer_ref = []
        self.loss = mloss
        self.dloss = mdloss
        self.activation = act
        self.dact = dact
        self.out_act = out
        self.dout_act = dout
        self.width = width
        self.optimizer = optimizer
        self.batch = batch
        self.sample_size = 0
        self.batch_size = batch_size
        self.iter_count = 0
        self.is_CN = CN
        self.screen = screen

        last = None
        for i in range(depth):
            if(depth == 1):
                n = Neuron(inputsize, classes)
                l = Layer(is_hidden = True, neurons = n, last_layer = last, activation = self.activation, loss = self.loss, dloss = self.dloss)
            elif(depth == 2 and i == 0):
                n = Neuron(inputsize, width[i+1])
                l = Layer(is_in = True, is_hidden = True, neurons = n, last_layer = last, activation = self.activation, loss = self.loss, dloss = self.dloss)
            elif(depth == 2 and i == 1):
                n = Neuron(width[i], classes)
                if(self.is_CN):
                    l = Layer(is_out = True, neurons = n, last_layer = last, activation = SF, loss = self.loss, dloss = self.dloss, use_CN = self.is_CN)
                else:
                    l = Layer(is_out = True, neurons = n, last_layer = last, activation = self.activation, loss = self.loss, dloss = self.dloss, use_CN = self.is_CN)
                self.final_layer = l
            else:
                if(i == 0):
                    n = Neuron(width[i], width[i + 1])
                    l = Layer(is_in = True, neurons = n, last_layer = last, activation = self.activation, loss = self.loss, dloss = self.dloss)
                elif(i == depth - 1):
                    n = Neuron(width[i], classes)
                    if(self.is_CN):
                        l = Layer(is_out = True, neurons = n, last_layer = last, activation = SF, loss = self.loss, dloss = self.dloss, use_CN = self.is_CN)
                    else:
                        l = Layer(is_hidden = True, neurons = n, last_layer = last, activation = self.out_act, loss = self.loss, dloss = self.dloss, use_CN = self.is_CN)
                    self.final_layer = l
                else:
                    n = Neuron(width[i], width[i + 1])
                    l = Layer(is_hidden = True, neurons = n, last_layer = last, activation = self.activation, loss = self.loss, dloss = self.dloss)
            if(i == 0):
                self.first_layer = l
                l.is_in = True
            if(i == depth - 1):
                l.is_out = True
                self.final_layer = l
            self.layer_ref.append(l)
            if(last):
                last.next_layer = l
                l.last_layer = last
            last = l

    def update_loss_acc(self, str_arr):
        tmp = ""
        for idx, p in enumerate(str_arr):
            tmp += p
            if(idx != len(str_arr) - 1):
                tmp += "\n"
        self.screen.set(tmp)

    def learn(self):
        tl = self.first_layer
        while(True):
            if(self.batch):
                offset = tl.batch_offset / self.iter_count
            else:
                offset = tl.offset
            if(self.optimizer == "momentum"):
                the_momentum = 0.8 * tl.last_momentum + self.lr * offset
                tl.neurons.weights = tl.neurons.weights - the_momentum
                tl.last_momentum = the_momentum
            elif(self.optimizer == "adam"):
                tl.adam_learned += 1
                m = 0.9 * tl.last_m + ( 1 - 0.9) * offset
                v = 0.999 * tl.last_v + (1 - 0.999) * np.square(offset)
                tl.last_m = m
                tl.last_v = v
                m_fixed = m / (1 - 0.9 ** tl.adam_learned)
                v_fixed = v / (1 - 0.999 ** tl.adam_learned)
                tl.neurons.weights = tl.neurons.weights - (self.lr * m_fixed / (np.sqrt(v_fixed) + 10**-8 ))
            elif(self.optimizer == "SGD"):
                tl.neurons.weights = tl.neurons.weights - self.lr * offset
            if(tl.next_layer is None):
                break
            else:
                tl = tl.next_layer


    def get_weights(self):
        l = self.first_layer
        tmp = []
        while(True):
            tmp.append(l.neurons.weights)
            if(l.next_layer is None):
                return tmp
            l = l.next_layer

    def train(self, inputs, target):
        for_print = []
        self.sample_size = len(inputs)
        if(self.max_batch < 0):
            self.max_batch = 20000
        iter_size = 1
        if(self.batch_size is not None):
            iter_size = int(len(inputs) / self.batch_size) + 1
            left_size = len(inputs) % self.batch_size
            if(iter_size < 2 or self.optimizer == "SGD"):
                iter_size = 1
        else:
            self.batch_size = len(inputs)
        for bt in range(self.max_batch):
            self.end_outputs = []
            for b in range(iter_size):
                if(b == iter_size - 1 and self.batch):
                    size_in_batch = left_size
                else:
                    size_in_batch = self.batch_size
                for i in range(size_in_batch):
                    tmp_l = self.first_layer
                    while(True):
                        tmp_l.forward(inputs[self.batch_size * b + i])
                        if(tmp_l.next_layer is not None):
                            tmp_l = tmp_l.next_layer
                        else:
                            break
                    tmp_l = self.final_layer
                    self.end_outputs.append(tmp_l.outputs)

                    while(True):
                        tmp_l.backprop(target = target[self.batch_size * b + i], lr = self.lr, dloss = self.dloss, dact = self.dact, dout_act = self.dout_act)
                        if(tmp_l.last_layer is not None):
                            tmp_l = tmp_l.last_layer
                        else:
                            break
                    if(not self.batch):
                        self.learn()
                    else:
                        self.iter_count += 1
                        self.record_batch()
                
                if(self.batch and self.iter_count != 0):
                    self.learn()
                    self.clear_batch_record()
            acc_tmp = self.accuracy(inputs, target)
            loss_tmp = self.loss_value(target, np.asarray(self.end_outputs))
            tmp = "loss: %.6f acc: %.6f" % (loss_tmp, acc_tmp)
            for_print.append(tmp)
            if(len(for_print) > 10):
                for_print = for_print[1:]
            print(tmp)
            if(acc_tmp >= self.target_acc):
                self.final_aver_loss = loss_tmp
                return bt + 1
        
        self.final_aver_loss = self.loss_value(target, np.asarray(self.end_outputs))
        return self.max_batch
        
    def accuracy(self, inps, targets):
        count = targets.shape[0]
        gotcha = 0
        for i in range(count):
            ins = inps[i]
            for dp, l in enumerate(self.layer_ref):
                ins = l.activation(ins.dot(l.neurons.weights))
            #print(ins, targets[i])
            if(np.argmax(ins) == np.argmax(targets[i])):
                gotcha += 1
        return float(gotcha / count)

    def pred_2d(self, inps_X, inps_Y, use_bias):
        count = inps_X.shape[0]
        gotcha = 0
        pred = []
        y = inps_Y
        x = inps_X
        for i in range(count):
            tmp = []
            for k in range(count):
                ins_x = x[i][k]
                ins_y = y[i][k]
                if(use_bias == 1):
                    mins = np.asarray([1, ins_x, ins_y])
                else:
                    mins = np.asarray([ins_x, ins_y])
                for dp, l in enumerate(self.layer_ref):
                    mins = l.activation(mins.dot(l.neurons.weights))

                tmp.append(np.argmax(mins))
            pred.append(tmp)
        return np.asarray(pred)

    def record_batch(self):
        tmp_l = self.first_layer
        while(tmp_l is not None):
            tmp_l.batch_offset = tmp_l.batch_offset + tmp_l.offset
            tmp_l = tmp_l.next_layer

    def clear_epoch_record(self):
        tmp_l = self.first_layer
        while(tmp_l is not None):
            tmp_l.last_m = 0
            tmp_l.last_v = 0
            tmp_l.last_momentum = 0
            tmp_l.adam_learned = 0
            tmp_l = tmp_l.next_layer

    def clear_batch_record(self):
        tmp_l = self.first_layer
        while(tmp_l is not None):
            self.iter_count = 0
            tmp_l.batch_offset = 0
            tmp_l = tmp_l.next_layer


    def loss_value(self, target, outs):
        l = self.loss(t = target, o = outs)
        loss = np.sum(l) / len(l)
        return loss