"""
   seq_models.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import math


class SRN_model(nn.Module):
    def __init__(self, num_input, num_hid, num_out, batch_size=1):
        super().__init__()
        self.num_hid = num_hid
        self.batch_size = batch_size
        self.H0= nn.Parameter(torch.Tensor(num_hid))
        self.W = nn.Parameter(torch.Tensor(num_input, num_hid))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid))
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid))
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        self.out_bias = nn.Parameter(torch.Tensor(num_out))

    def init_hidden(self):
        H0 = torch.tanh(self.H0)
        return(H0.unsqueeze(0).expand(self.batch_size,-1))
 
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t = self.init_hidden().to(x.device)
        else:
            h_t = init_states
            
        for t in range(seq_size):
            x_t = x[:, t, :]
            c_t = x_t @ self.W + h_t @ self.U + self.hid_bias
            h_t = torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from (sequence, batch, feature)
        #           to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        output = hidden_seq @ self.V + self.out_bias
        return hidden_seq, output

class LSTM_model(nn.Module):
    def __init__(self,num_input,num_hid,num_out,batch_size=1,num_layers=1):
        super().__init__()
        # num_input = 7
        self.num_hid = num_hid   # 2
        self.batch_size = batch_size   # 5
        self.num_layers = num_layers      # 1
        # 7 hidden layer
        # And we need:
        # `input gate`: i_t (sigmoid layer)
        # `forget gate`: f_t(the first sigmoid layer)
        # `cell gate(or cell state, a conveyor belt)`:
        #  - ⊕: combine the information
        #  - ⊗: (tensor product) decide which one will be screened in
        # `out_put gate`:  decide how much of the cell state should be revealed to the next hidden state

        # self.W would be a 7x8 tensor.[num_input * (num_hid * 4 gates)]
        # Each row corresponds to an input feature,
        # and each group of four columns corresponds to the weights for a hidden unit's four gates.
        self.W = nn.Parameter(torch.Tensor(num_input, num_hid * 4))
        self.U = nn.Parameter(torch.Tensor(num_hid, num_hid * 4))

        # tow hidden layer: (2 * 4) bias
        self.hid_bias = nn.Parameter(torch.Tensor(num_hid * 4))
        # only one output layer:
        # perform a linear transformation from the hidden state to the output
        # create weight matrix V for the output units
        self.V = nn.Parameter(torch.Tensor(num_hid, num_out))
        # bias of output layer
        self.out_bias = nn.Parameter(torch.Tensor(num_out))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.num_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self):
        return(torch.zeros(self.num_layers, self.batch_size, self.num_hid),
               torch.zeros(self.num_layers, self.batch_size, self.num_hid))

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        # store parameters of each layer
        C_t = []
        I_t = []
        F_t = []
        O_t = []
        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size,self.num_hid).to(x.device), 
                        torch.zeros(batch_size,self.num_hid).to(x.device))
        else:
            h_t, c_t = init_states

        NH = self.num_hid
        for t in range(seq_size):
            # selecting the input at the current time step.
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.hid_bias
            i_t, f_t, g_t, o_t = (
                # input gate
                torch.sigmoid(gates[:, :NH]),
                # forget gate
                torch.sigmoid(gates[:, NH:NH*2]),
                # new values(this is the C_t, that could be added to the state)
                torch.tanh(gates[:, NH*2:NH*3]),
                # output gate
                # put the cell state through tanh (to push the values to be between −1 and 1)
                # and multiply it by the output of the sigmoid gate,
                # so that we only output the parts we decided to.
                # h_t = o_t * tanh(C_t)
                torch.sigmoid(gates[:, NH*3:]),
            )
            # c_t means:
            #  drop the information about the old subject’s gender
            #  and add the new information
            c_t = f_t * c_t + i_t * g_t
            # h_t = o_t * tanh(C_t)
            # put the cell state into tanh,
            # so that we only output the parts we decided do.
            h_t = o_t * torch.tanh(c_t)
            C_t.append(c_t)
            F_t.append(f_t)
            O_t.append(o_t)
            I_t.append(i_t)
            hidden_seq.append(h_t.unsqueeze(0))
        # concatenate the given sequence in the dimension = 0
        #     >>> x
        #     tensor([[ 0.6580, -1.0969, -0.4614],
        #             [-0.1034, -0.5790,  0.1497]])
        #     >>> torch.cat((x, x, x), 0)
        #     tensor([[ 0.6580, -1.0969, -0.4614],
        #             [-0.1034, -0.5790,  0.1497],
        #             [ 0.6580, -1.0969, -0.4614],
        #             [-0.1034, -0.5790,  0.1497],
        #             [ 0.6580, -1.0969, -0.4614],
        #             [-0.1034, -0.5790,  0.1497]])
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from (sequence, batch, feature)
        #           to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0,1).contiguous()
        # convert to properbility
        output = hidden_seq @ self.V + self.out_bias
        return hidden_seq, output, C_t, F_t, O_t, I_t
