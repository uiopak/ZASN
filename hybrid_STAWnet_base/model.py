import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class graphattention(nn.Module):
    def __init__(self,c_in,c_out,dropout,d=16, emb_length=0, aptonly=False, noapt=False):
        super(graphattention,self).__init__()
        self.d = d
        self.aptonly = aptonly
        self.noapt = noapt
        self.mlp = linear(c_in*2,c_out)
        self.dropout = dropout
        self.emb_length = emb_length
        if aptonly:
            self.qm = FC(self.emb_length, d)  # query matrix
            self.km = FC(self.emb_length, d)  # key matrix
        elif noapt:
            self.qm = FC(c_in, d)  # query matrix
            self.km = FC(c_in, d)  # key matrix
        else:
            self.qm = FC(c_in + self.emb_length, d)  # query matrix
            self.km = FC(c_in + self.emb_length, d)  # key matrix

    def forward(self,x,embedding):
        # x: [batch_size, D, nodes, time_step]
        # embedding = [10, num_nodes]
        out = [x]

        embedding = embedding.repeat((x.shape[0], x.shape[-1], 1, 1)) # embedding = [batch_size, time_step, 10, num_nodes]
        embedding = embedding.permute(0,2,3,1) # embedding = [batch_size, 16, num_nodes, time_step]

        if self.aptonly:
            x_embedding = embedding
            query = self.qm(x_embedding).permute(0, 3, 2, 1)
            key = self.km(x_embedding).permute(0, 3, 2, 1)  #
            # value = self.vm(x)
            attention = torch.matmul(query,key.permute(0, 1, 3, 2))  # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        elif self.noapt:
            x_embedding = x
            query = self.qm(x_embedding).permute(0, 3, 2, 1)  # query=[batch_size, time_step, num_nodes, d]
            key = self.km(x_embedding).permute(0, 3, 2, 1)  # key=[batch_size, time_step, num_nodes, d]
            attention = torch.matmul(query,key.permute(0, 1, 3, 2))  # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        else:
            x_embedding = torch.cat([x,embedding], axis=1) # x_embedding=[batch_size, D+10, num_nodes, time_step]
            query = self.qm(x_embedding).permute(0,3,2,1) # query=[batch_size, time_step, num_nodes, d]
            key = self.km(x_embedding).permute(0,3,2,1) # key=[batch_size, time_step, num_nodes, d]
            # query = F.relu(query)
            # key = F.relu(key)
            attention = torch.matmul(query, key.permute(0,1,3,2)) # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d**0.5)
            attention = F.softmax(attention, dim=-1)

        x = torch.matmul(x.permute(0,3,1,2), attention).permute(0,2,3,1)
        out.append(x)

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h, 0#attention


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class FC(nn.Module):
    def __init__(self,c_in,c_out):
        super(FC,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
    
class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class graph_stawnet(nn.Module):
    def __init__(self, device, num_nodes, gat_blocks, gcn_blocks,
                 dropout=0.3, supports=None, gat_bool=True, gcn_bool=True,
                 addaptadj=True, aptonly=False, noapt=False, aptinit=None,
                 in_dim=2,out_dim=12, residual_channels=32,
                 dilation_channels=32,skip_channels=256,end_channels=512,
                 kernel_size=2,layers=2,emb_length=16):
        super(graph_stawnet, self).__init__()
        self.dropout = dropout
        self.gat_blocks = gat_blocks
        self.gcn_blocks = gcn_blocks
        self.layers = layers
        self.gat_bool = gat_bool
        self.gcn_bool = gcn_bool
        self.aptonly = aptonly
        self.noapt = noapt
        self.addaptadj = addaptadj
        self.emb_length = emb_length
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gat = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1
        
        ## GRAPH-WAVENET 
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
        ## STAWNET 
        if gat_bool and addaptadj:
            self.gat_supports = nn.Parameter(torch.randn(self.emb_length, num_nodes).to(device), requires_grad=True).to(device)

        for b in range(self.gat_blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gat_bool:
                    self.gat.append(graphattention(dilation_channels,residual_channels,dropout, emb_length=emb_length, aptonly=aptonly, noapt=noapt))
                    
        for b in range(self.gcn_blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))


        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
       
        for i in range(self.gat_blocks * self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gat_bool:
                if self.addaptadj:
                    x, att = self.gat[i](x, self.gat_supports)
                    # attentions.append(att.cpu().detach().numpy()[:,0,:,:]) # record every attention matrix in all layers

            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)



        ## GRAPH-WAVENET 
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
            
        # WaveNet layers
        for i in range(self.gcn_blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)
            
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # final_attention = np.array(attentions[:-1]) # because last attention isn't used
        return x, 0#final_attention





