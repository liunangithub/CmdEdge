
import torch.nn.functional as F
import torch as th
from torch_scatter import scatter_mean


class Classfier(th.nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes):
        super(Classfier, self).__init__()
        self.linear_one = th.nn.Linear(5000 * 2, 2 * hid_feats)   #(10000,128)
        self.linear_two = th.nn.Linear(2 * hid_feats, hid_feats)  #(128,64)
        self.linear_three = th.nn.Linear(hid_feats, hid_feats)     #(192,64)

        self.linear_transform = th.nn.Linear(hid_feats * 2, 4)     #(128,4)
        self.prelu = th.nn.PReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, th.nn.Linear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, embed, data):
        ori = scatter_mean(data.x, data.batch, dim=0)
        root = data.x[data.rootindex]
        ori = th.cat((ori, root), dim=1)
        ori = self.linear_one(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)
        ori = self.linear_two(ori)
        ori = F.dropout(input=ori, p=0.5, training=self.training)
        ori = self.prelu(ori)

        x = scatter_mean(embed, data.batch, dim=0)      #(128,64)
        x = self.linear_three(x)                        #(64,64)
        x = F.dropout(input=x, p=0.5, training=self.training)
        x = self.prelu(x)

        out = th.cat((x, ori), dim=1)            #(128,128)
        out = self.linear_transform(out)
        x = F.log_softmax(out, dim=1)
        return x
