import tkinter as tk

from tkinter import filedialog
import math
import os
import numpy as np

import traceback

import matplotlib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import time

import model
add1_x = lambda x: np.append(x, [[1]] * len(x) , axis = 1)

epsilon = lambda x: np.max(x, sys.float_info.epsilon)

loss_square = lambda t, o : np.square( t - o )
dloss_square = lambda t,o : -2 * t + 2 * o

LF = lambda x: 1 / (1 + np.exp(-x))
dLF = lambda x: LF(x) * (1 - LF(x))

SF = lambda x : np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
diSF = lambda x: SF(x) * (1 - SF(x))

SSF = lambda x : np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

loss = lambda t, o :  - ( t * np.log(np.maximum(o , 2.2250738585072014e-308)))
dloss = lambda t, o :  - ((t / o) + (1 - t) / (1 - o))

def relu(x):
    try:
        exx = np.exp(x)
    except Exception as e:
        if(e.args[0] == "overflow encountered in exp"):
            exx = np.inf
        else:
            exx = 0
        pass
    return (x >= 0) * x + (x < 0) * (0.5 * (exx - 1 ))


#relu = lambda x : (x >= 0) * x + (x < 0) * (0.5 * (np.exp(x) - 1 ))
#relu = lambda x : (x)
drelu = lambda x : (x >= 0) * 1 + (x < 0) * (relu(x) + 0.5)
act_equal = lambda x : x
deq = lambda x : 1
add1_x = lambda x: np.append(x, [[1]] * len(x) , axis = 1)
        
def draw_point(fn):
    plt.cla()
    df = open(fn)
    fl = df.readline()
    fl = list(fl.split())
    dim = len(fl) - 1
    root.dim = dim
    data = np.array([fl[:-1]])
    label = np.array([fl[-1]])
    for line in df:
        tmp_list = list(line.split())
        nptmp = np.array([ tmp_list[:-1] ])
        data = np.append(data, np.array([ tmp_list[:-1] ]), axis = 0)
        label = np.append(label, [tmp_list[-1]])
    root.samples.set(str(data.shape[0]))
    label = label.astype(np.float).astype(np.int)
    data = data.astype(np.float)
    root.input = [data, label]
    if(root.fig is not None):
        root.fig.clf()
        root.fig.clear()

    try:
        root.canvas.get_tk_widget().destroy()
    except:
        pass

    if(dim == 2):
        try:
            root.canvas.get_tk_widget().destroy()
        except:
            pass
        root.filestat.set(root.filename.get() + ": show 2D graph")
        root.fig = Figure(figsize = (75,75), dpi = 100)
        root.emb = root.fig.add_subplot(111)
        fixed_ticks = range(int(np.floor(np.amin(data))), int(np.ceil(np.amax(data))))
        plt.xticks(fixed_ticks)
        plt.yticks(fixed_ticks)
        #c = [cmaps[x] for x in root.input[1]]
        root.emb.scatter(data[:,0], data[:,1], c = [cmaps[x]  for x in root.input[1]], s = 2)
        root.data = data
        root.canvas = FigureCanvasTkAgg(root.fig, dfm)

        try:
            root.canvas.draw()
        except Exception as e:
            root.error_msg.set(e.__class__.__name__ + "\n" + "Out of memory\n plz restart" )
            error_class = e.__class__.__name__
            detail = e.args[0]
            cl, exc, tb = sys.exc_info()
            lastCallStack = traceback.extract_tb(tb)[-1]
            fileName = lastCallStack[0]
            lineNum = lastCallStack[1]
            funcName = lastCallStack[2]
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(errMsg)
            root.sb.config(state = "normal")
            return
        
        wd = root.canvas.get_tk_widget()
        wd.config(height="500", width="500")
        wd.pack(side = 'top')

    elif(dim == 3):
        try:
            root.canvas.get_tk_widget().destroy()
        except:
            pass
        root.filestat.set(root.filename.get() + ": show 3D graph")
        root.fig = Figure(figsize = (75,75), dpi = 100)
        root.emb = root.fig.add_subplot(111, projection='3d')
        fixed_ticks = range(int(np.floor(np.amin(data))), int(np.ceil(np.amax(data))))

        root.emb.set_xticks(fixed_ticks)
        root.emb.set_yticks(fixed_ticks)
        root.emb.set_zticks(fixed_ticks)
        root.emb.set_xlabel('X')
        root.emb.set_ylabel('Y')
        root.emb.set_zlabel('Z')
        root.emb.scatter(data[:,0], data[:,1], data[:,2], c=[cmaps[x] for x in root.input[1]], s = 2)
        root.canvas = FigureCanvasTkAgg(root.fig, dfm)
        root.canvas.draw()
        wd = root.canvas.get_tk_widget()
        wd.config(height="510", width="510")
        wd.pack(side = 'top')
        
    else:
        root.filestat.set(root.filename.get() + ": dims are greater than 3")
    root.sb.config(state = "normal")


def fill_colors(limit_x, limit_y, m, use_bias):

        root.canvas.get_tk_widget().destroy()
        root.fig.clf()
        root.fig.clear()
        root.fig = Figure(figsize = (75,75), dpi = 100)
        root.emb = root.fig.add_subplot(111)
        #c = [cmaps[x] for x in root.input[1]]
        root.emb.scatter(root.data[:,0], root.data[:,1], c = [cmaps[x] for x in root.input[1]], s = 2)
        px, py = np.meshgrid(np.linspace(limit_x[0], limit_x[1], 200), np.linspace(limit_y[0], limit_y[1],200))
        root.emb.contourf(px, py, m.pred_2d(px, py, use_bias), alpha=0.5)
        root.canvas = FigureCanvasTkAgg(root.fig, dfm)
        root.canvas.draw()
        wd = root.canvas.get_tk_widget()
        wd.config(height="500", width="500")
        wd.pack(side = 'top')


def label_encode(y):
    classes = len(np.unique(y))

    label = np.zeros((len(y), classes))

    label_set = np.unique(y)

    for order, l in enumerate(y):
        label[order][np.where(label_set == l)] = 1

    return label

def CN_force_SF():
    if(root.selected_out.get() != "softmax"):
        root.selected_out.set("softmax")

    if(root.loss_act.get() == "CN"):
        root.relu_out.config(state = 'disable')
        root.sig_out.config(state = 'disable')
        root.eq_out.config(state = 'disable')
    else:
        root.relu_out.config(state = 'normal')
        root.sig_out.config(state = 'normal')
        root.eq_out.config(state = 'normal')        





def get_file():
    path = filedialog.askopenfilename(initialdir = root.initdir, title = "Select file", filetypes=[("Txt Files","*.txt")])
    root.path.set(os.path.basename(path))
    root.initdir = os.path.dirname(path)
    root.filename.set(os.path.basename(path))
    draw_point(path)

def start():
    root.error_msg.set("If exception raises:")
    root.warning.set("")
    root.sb.config(state = "disabled")

    LR = float(root.lr_input.get())
    MB = int(root.max_batch.get().split()[0])
    if(len(root.max_batch.get().split()) == 2):
        BS = int(root.max_batch.get().split()[1])
    else:
        BS = None
    TACC = float(root.target_acc.get())
    DP = int(root.dp_input.get())
    x_p = (np.max(root.input[0][:, 0]) - np.min(root.input[0][:, 0])) * 0.1
    y_p = (np.max(root.input[0][:, 1]) - np.min(root.input[0][:, 1])) * 0.1
    x_limit = [np.min(root.input[0][:, 0]) - x_p , np.max(root.input[0][:, 0]) + x_p]
    y_limit = [np.min(root.input[0][:, 1]) - y_p , np.max(root.input[0][:, 1]) + y_p]

    if(root.Do_not_test.get() == 0):
        concat = np.append(root.input[0], np.reshape(root.input[1], (root.input[1].shape[0],1) ), axis = 1)
        np.random.shuffle(concat)
        if(root.bias.get() == 1):
            bias = np.full((concat.shape[0], 1), fill_value = 1)
            concat = np.concatenate((bias, concat), axis = 1)

        test_len = int(len(concat)/3)
        testing_data = concat[:test_len]
        training_data = concat[test_len:]

        testing_y = testing_data[:, -1]
        testing_x = testing_data[:, :-1]

        training_y = training_data[:, -1]
        training_x = training_data[:, :-1]
        
        training_y = label_encode(training_y)
        testing_y = label_encode(testing_y)
    else:
        concat = np.append(root.input[0], np.reshape(root.input[1], (root.input[1].shape[0],1) ), axis = 1)
        if(root.bias.get() == 1):
            bias = np.full((concat.shape[0], 1), fill_value = 1)
            concat = np.concatenate((bias, concat), axis = 1)

        testing_y = concat[:, -1]
        testing_x = concat[:, :-1]

        training_y = concat[:, -1]
        training_x = concat[:, :-1]

        training_y = label_encode(training_y)
        testing_y = label_encode(testing_y)

    classes = len(np.unique(root.input[1]))

    act_str = root.selected_act.get()

    use_CN = False

    if(act_str == "sigmoid"):
        act = LF
        dact = dLF
    elif(act_str == "relu"):
        act = relu
        dact = drelu
    elif(act_str == "equal"):
        act = act_equal
        dact = deq
    loss_str = root.loss_act.get()
    if(loss_str == "MSE"):
        ploss = loss_square
        dploss = dloss_square
    elif(loss_str == "CN"):
        ploss = loss
        dploss = dloss
        use_CN = True

    out_act_str = root.selected_out.get()
    if(out_act_str == "equal"):
        out_act = lambda x : x
        dout_act = lambda x : 1
    elif(out_act_str == "softmax"):
        out_act = SF
        dout_act = diSF
    elif(act_str == "sigmoid"):
        out_act = LF
        dout_act = dLF
    elif(act_str == "relu"):
        out_act = relu
        dout_act = drelu
    

    width = []

    width.append(np.asarray(training_x).shape[1])
    tmp_width = list(map(int, root.width.get().split()))
    tmp_width = tmp_width[:DP]
    width = width + list(map(int, root.width.get().split()))


    if(root.optimizer.get() == "SGD"):
        select_batch = False
    else:
        select_batch = True

    m = model.model(DP, np.asarray(training_x).shape[1]\
        , classes\
        , LR\
        , MB\
        , TACC\
        , width\
        , act = act\
        , dact = dact\
        , mloss = ploss\
        , mdloss = dploss\
        , out = out_act\
        , dout = dout_act\
        , optimizer = root.optimizer.get()\
        , batch = select_batch\
        , batch_size = BS\
        , CN = use_CN)

    try:
        root.trained.set(str(m.train(training_x, training_y)))
    except Exception as e:
        root.error_msg.set(e.__class__.__name__+ "\n" + "please set lower lr.")
        error_class = e.__class__.__name__
        detail = e.args[0]
        cl, exc, tb = sys.exc_info()
        lastCallStack = traceback.extract_tb(tb)[-1]
        fileName = lastCallStack[0]
        lineNum = lastCallStack[1]
        funcName = lastCallStack[2]
        errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        print(errMsg)
        root.sb.config(state = "normal")
        return

    st = m.get_weights()
    pw = ""
    for w1 in st:
        for w2 in w1:
            pw = pw + str(w2)
        pw = pw + "\n"
    root.weights.set(pw)
    tsacc = m.accuracy(testing_x, testing_y)
    root.ts_acc.set(tsacc)
    root.tr_acc.set(m.accuracy(training_x, training_y))
    root.floss.set(str(m.final_aver_loss))

    if(root.dim == 2):
        try:
            fill_colors(x_limit, y_limit, m, root.bias.get())
        except Exception as e:
            root.error_msg.set(e.__class__.__name__ + "\n" + "Out of memory\n plz restart" )
            error_class = e.__class__.__name__
            detail = e.args[0]
            cl, exc, tb = sys.exc_info()
            lastCallStack = traceback.extract_tb(tb)[-1]
            fileName = lastCallStack[0]
            lineNum = lastCallStack[1]
            funcName = lastCallStack[2]
            errMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(errMsg)
            root.sb.config(state = "normal")
            return
#    print(str(m.all_loss_last))

    
    root.sb.config(state = "normal")



if(__name__ == '__main__'):
    np.seterr(over='raise')
    matplotlib.use("TkAgg")
    cmaps = ['orange', 'red', 'green', 'blue', 'black', 'yellow']

    root = tk.Tk()
    root.title('Why GUIs bully me?')
    root.geometry('1300x720')
    root.filename = tk.StringVar()
    root.path = tk.StringVar()
    root.filename.set("Selecting file...")
    root.initdir = "/"
    root.filestat = tk.StringVar()
    root.weights = tk.StringVar()
    root.selected_act = tk.StringVar(None, "relu")
    root.loss_act = tk.StringVar(None, "MSE")
    root.selected_out = tk.StringVar(None, "sigmoid")
    root.Do_not_test = tk.IntVar()
    root.bias = tk.IntVar(None, 1)
    root.trained = tk.StringVar()
    root.optimizer = tk.StringVar(None, "adam")
    root.fig = None
    root.error_msg = tk.StringVar(None, "If exception raises:")
    root.samples = tk.StringVar()

    fm = tk.Frame(root)
    fm.pack()
    lfm = tk.Frame(fm, width = 200, height = 650)
    mfm = tk.Frame(fm, width = 1000, height = 650)
    cfm = tk.Canvas(mfm, width = 300, height = 100)
    rfm = tk.Frame(mfm, width = 100, height = 100)
    rtm = tk.Frame(mfm, width = 100, height = 100)
    rrfm = tk.Frame(mfm, width = 800, height = 100)
    dfm = tk.Frame(fm, width = 800, height = 650)
    errfm = tk.Frame(fm, width = 200, height = 650)

    lfm.pack(side = 'left', padx = 8, pady = 0)
    lfm.pack_propagate(0)

    mfm.pack(anchor = 'ne', padx = 8, pady = 0)

    cfm.pack(side = 'left', padx = 4, pady = 0)
    rfm.pack(side = 'left', padx = 4, pady = 0)
    rtm.pack(side = 'left', padx = 4, pady = 0)

    vbar = tk.Scrollbar(cfm, orient = tk.VERTICAL)
    vbar.pack(side = 'right')
    vbar.configure(command = cfm.yview)
    cfm.config(yscrollcommand=vbar.set)

    rfm.pack_propagate(0)
    rtm.pack_propagate(0)

    rrfm.pack(side = 'right', padx = 4, pady = 0)
    rrfm.pack_propagate(0)

    errfm.pack(side = 'left', pady = 2, padx = 4)
    errfm.pack_propagate(0)

    dfm.pack(side = 'top', pady = 2, padx = 4)
    dfm.pack_propagate(0)
    



    filelabel = tk.Label(dfm, textvariable = root.filestat)
    filelabel.pack(anchor="n")

    tk.Button(lfm, text = "Select file of data", command = get_file).pack(anchor = 'w')
    tk.Label(lfm, textvariable = root.filename).pack(anchor = "w")


    lr_text = tk.Label(lfm, text = 'Leaning rate: ', font = ('Arial', 9))
    lr_text.pack(anchor = "w")

    root.lr_input = tk.Entry(lfm, show = "", width = 8)
    root.lr_input.insert(tk.END, "0.03")
    root.lr_input.pack(anchor = "w")

    dp_text = tk.Label(lfm, text = 'Depth: ', font = ('Arial', 9))
    dp_text.pack(anchor = "w")

    root.dp_input = tk.Entry(lfm, show = "", width = 5)
    root.dp_input.insert(tk.END, "4")
    root.dp_input.pack(anchor = "w")

    con_text = tk.Label(lfm, text = 'Convergence condition: ', font = ('Arial', 9))
    con_text.pack(anchor = "w")

    tk.Label(lfm, text = '  Max epochs and batch size: ', font = ('Arial', 8)).pack(anchor = "w")
    root.max_batch = tk.Entry(lfm, show = "", width = 15)
    root.max_batch.insert(tk.END, "200 10")
    root.max_batch.pack(anchor = "w", padx = 8)

    tk.Label(lfm, text = '  Target Accuracy: ', font = ('Arial', 8)).pack(anchor = "w")
    root.target_acc = tk.Entry(lfm, show = "", width=5)
    root.target_acc.insert(tk.END, "0.8")
    root.target_acc.pack(anchor = "w", padx = 8)

    tk.Label(lfm, text = '  the width per hidden layer: ', font = ('Arial', 8)).pack(anchor = "w")
    root.width = tk.Entry(lfm, show = "", width = 15)
    root.width.insert(tk.END, "6 6 3")
    root.width.pack(anchor = "w", padx = 8)

    root.sb = tk.Button(lfm, text = "START", command = start, state = "disabled")
    root.sb.pack(anchor = 'center', pady = 0)

    root.warning = tk.StringVar()
    tk.Label(lfm, textvariable = root.warning, font = ('Arial', 7), fg = 'red').pack(anchor = "w", pady=2)

    tk.Radiobutton(lfm, text = 'sigmoid', variable = root.selected_act, value = 'sigmoid').pack(anchor = "w", padx = 2)
    tk.Radiobutton(lfm, text = 'relu', variable = root.selected_act, value = 'relu').pack(anchor = "w", padx = 2)
    tk.Radiobutton(lfm, text = 'equal', variable = root.selected_act, value = 'equal').pack(anchor = "w", padx = 2)

    loss_select = tk.Frame(lfm)
    loss_select.pack(anchor = "w", padx = 4, pady = 0)
    tk.Radiobutton(loss_select, text = 'MSE', variable = root.loss_act, value = 'MSE', command = CN_force_SF).pack(anchor = "w", padx = 2)
    tk.Radiobutton(loss_select, text = 'CN', variable = root.loss_act, value = 'CN', command = CN_force_SF).pack(anchor = "w", padx = 2)

    out_select = tk.Frame(lfm)
    out_select.pack(anchor = "w", padx = 6, pady = 0)
    root.eq_out = tk.Radiobutton(loss_select, text = 'out_equal', variable = root.selected_out, value = 'equal')
    root.eq_out.pack(anchor = "w", padx = 4)
    root.sf_out = tk.Radiobutton(loss_select, text = 'out_softmax', variable = root.selected_out, value = 'softmax')
    root.sf_out.pack(anchor = "w", padx = 4)
    root.sig_out = tk.Radiobutton(loss_select, text = 'out_sigmoid', variable = root.selected_out, value = 'sigmoid')
    root.sig_out.pack(anchor = "w", padx = 4)
    root.relu_out = tk.Radiobutton(loss_select, text = 'out_relu', variable = root.selected_out, value = 'relu')
    root.relu_out.pack(anchor = "w", padx = 4)

    test_or_not = tk.Frame(lfm)
    test_or_not.pack(anchor = "w", padx = 8, pady = 0)
    tk.Checkbutton(test_or_not, text='Do not test', variable = root.Do_not_test, onvalue=1, offvalue=0).pack(anchor = "w", padx = 6)
    momentum = tk.Frame(lfm)
    momentum.pack(anchor = "w", padx = 8, pady = 0)
    tk.Radiobutton(momentum, text='momentum', variable = root.optimizer, value = "momentum").pack(anchor = "w", padx = 6)
    tk.Radiobutton(momentum, text='adam', variable = root.optimizer, value = "adam").pack(anchor = "w", padx = 6)
    tk.Radiobutton(momentum, text='SGD', variable = root.optimizer, value = "SGD").pack(anchor = "w", padx = 6)
    tk.Checkbutton(lfm, text='bias(threshold)', variable = root.bias, onvalue=1, offvalue=0).pack(anchor = "w", padx = 6)

    root.tr_acc = tk.StringVar()
    root.tr_acc.set("Wait for training...")
    root.ts_acc = tk.StringVar()
    root.ts_acc.set("Wait for training...")
    root.floss = tk.StringVar()
    root.floss.set("Wait for training...")

    tk.Label(rfm, text = 'training acc: ', font = ('Arial', 7)).pack(anchor = "w")
    tn_acc_text = tk.Label(rfm, textvariable = root.tr_acc, font = ('Arial', 6)).pack(anchor = "w", padx = 2)
    tk.Label(rfm, text = 'testing acc: ', font = ('Arial', 7)).pack(anchor = "w")
    ts_acc_text = tk.Label(rfm, textvariable = root.ts_acc, font = ('Arial', 6)).pack(anchor = "w", padx = 2)
    tk.Label(rfm, text = 'final loss: ', font = ('Arial', 7)).pack(anchor = "w")
    floss_text = tk.Label(rfm, textvariable = root.floss, font = ('Arial', 6)).pack(anchor = "w", padx = 2)
    
    tk.Label(rtm, text = 'training times: ', font = ('Arial', 7)).pack(anchor = "w")
    trained_text = tk.Label(rtm, textvariable = root.trained, font = ('Arial', 6)).pack(anchor = "w", padx = 2)
    tk.Label(rtm, text = 'samples: ', font = ('Arial', 7)).pack(anchor = "w")
    samples_text = tk.Label(rtm, textvariable = root.samples, font = ('Arial', 6)).pack(anchor = "w", padx = 2)

    tk.Label(errfm, textvariable = root.error_msg, font = ('Arial', 7)).pack(anchor = "w")
    


    tk.Label(rrfm, text = 'Weights:' , font = ('Arial', 10)).pack(anchor = "w", pady = 0)
    weights_text = tk.Label(rrfm, textvariable = root.weights, font = ('Arial', 7)).pack(anchor = "w", padx = 2)

    tk.mainloop()
