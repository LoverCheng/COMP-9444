"""
   seq_train.py
   COMP9444, CSE, UNSW
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from seq_models import SRN_model, LSTM_model
from reber import lang_reber
from anbn import lang_anbn


parser = argparse.ArgumentParser()
# language options
parser.add_argument('--lang', type=str, default='reber', help='reber, anbn or anbncn')
parser.add_argument('--embed', type=bool, default=False, help='embedded or not (reber)')
parser.add_argument('--length', type=int, default=0, help='min (reber) or max (anbn)')
# network options
parser.add_argument('--model', type=str, default='srn', help='srn or lstm')
parser.add_argument('--hid', type=int, default=0, help='number of hidden units')
# optimizer options
parser.add_argument('--optim', type=str, default='adam', help='sgd or adam')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--mom', type=float, default=0, help='momentum (srn)')
parser.add_argument('--init', type=float, default=0.001, help='initial weight size (srn)')
# training options
parser.add_argument('--epoch', type=int, default=0, help='number of training epochs (\'000s)')
parser.add_argument('--out_path', type=str, default='net', help='outputs path')
args = parser.parse_args()

if args.lang == 'reber':
    num_class = 7
    hid_default = 2
    epoch_default = 50
    lang = lang_reber(args.embed,args.length)
elif args.lang == 'anbn':
    num_class = 2
    hid_default = 2
    epoch_default = 100
    if args.length == 0:
        args.length = 8
    lang = lang_anbn(num_class,args.length)
elif args.lang == 'anbncn':
    num_class = 3
    hid_default = 3
    epoch_default = 200
    if args.length == 0:
        args.length = 8
    lang = lang_anbn(num_class,args.length)

if args.hid == 0:
    args.hid = hid_default

if args.epoch == 0:
    args.epoch = epoch_default
    
if args.model == 'srn':
    net = SRN_model(num_class,args.hid,num_class)
    for m in list(net.parameters()):
        m.data.normal_(0,args.init)
elif args.model == 'lstm':
    net = LSTM_model(num_class,args.hid,num_class)

if args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=0.00001)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.mom, weight_decay=0.00001)

loss_function = F.nll_loss

np.set_printoptions(suppress=True,precision=2,sign=' ')

C_t_mean_final = []
F_t_mean_final = []
O_t_mean_final = []
I_t_mean_final = []

for epoch in range((args.epoch*1000)+1):
    net.zero_grad()

    input, seq, target, state = lang.get_sequence()
    label  = seq[1:]

    net.init_hidden()
    # hidden, output = net(input)
    hidden, output, C_t, F_t, O_t, I_t = net(input)
    log_prob = F.log_softmax(output, dim=2)
    prob_out = torch.exp(log_prob)
    loss = F.nll_loss(log_prob.squeeze(), label.squeeze())
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:

        # Check accuracy during training
        with torch.no_grad():
            net.eval()

            input, seq, target, state = lang.get_sequence()
            label = seq[1:]
            
            net.init_hidden()
            # hidden, output = net(input)
            hidden, output, C_t, F_t, O_t, I_t = net(input)
            C_t_np = [tensor.numpy() for tensor in C_t]
            F_t_np = [tensor.numpy() for tensor in F_t]
            O_t_np = [tensor.numpy() for tensor in O_t]
            I_t_np = [tensor.numpy() for tensor in I_t]
            # Calculate means
            C_t_mean = np.mean([np.mean(arr) for arr in C_t_np])
            F_t_mean = np.mean([np.mean(arr) for arr in F_t_np])
            O_t_mean = np.mean([np.mean(arr) for arr in O_t_np])
            I_t_mean = np.mean([np.mean(arr) for arr in I_t_np])


            log_prob = F.log_softmax(output, dim=2)
            prob_out = torch.exp(log_prob)
            
            lang.print_outputs(epoch, seq, state, hidden, target, output)
            print("C_t:")
            print(C_t)
            print("F_t")
            print(F_t)
            print("O_t")
            print(O_t)
            print("I_t")
            print(I_t)
            sys.stdout.flush()

            C_t_mean_final.append(C_t_mean)
            F_t_mean_final.append(C_t_mean)
            O_t_mean_final.append(C_t_mean)
            I_t_mean_final.append(C_t_mean)
            net.train()

        if epoch % 10000 == 0:
            path = args.out_path+'/'
            torch.save(net.state_dict(),path+'%s_%s%d_%d.pth'
                       %(args.lang,args.model,args.hid,epoch/1000))

t= np.arange(len(C_t_mean_final))
plt.figure(figsize=(12, 8))
plt.plot(t, C_t_mean, label='C_t')
plt.plot(t, F_t_mean, label='F_t')
plt.plot(t, O_t_mean, label='O_t')
plt.plot(t, I_t_mean, label='I_t')
plt.xlabel('Timestep')
plt.ylabel('Mean Value')
plt.title('Mean Gate and Cell State Values Over Time')
plt.legend()
plt.show()