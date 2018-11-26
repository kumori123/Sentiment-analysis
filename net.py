from torch import nn
import torch.nn.utils.rnn as rnn
import torch
import torch.nn.functional as F
class Batch_Net(nn.Module):


    def __init__(self, in_dim, n_hidden_1, out_dim):

        super(Batch_Net, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))

        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, out_dim))



    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        return x



class SentimentNet(nn.Module):
    def __init__(self, embed_size, num_hiddens, num_layers,
                 bidirectional,labels,**kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0, batch_first= True)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 1, labels)

    def forward(self, inputs):
        inputs = inputs.permute([1,0,2])
        #embed_input_x_packed = rnn.pack_padded_sequence(input, sentence_lens, batch_first=True)
        #encoder_outputs_packed, (h_last, c_last) = self.encoder(embed_input_x_packed)
        #encoder_output,_ = rnn.pack_padded_sequence(encoder_outputs_packed,batch_first=T)
        states, hidden = self.encoder(inputs)
        encoding = torch.cat([states[0]], dim=1)
        #encoding = states[-1]
        outputs = self.decoder(encoding)
        return outputs