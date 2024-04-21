import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from utils import *

class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    """ a network in network layer (1x1 CONV)
    """
    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                    shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), # pad left
                                  int((filter_size[1] - 1) / 2), # pad right
                                  filter_size[0] - 1,            # pad top
                                  0) )                           # pad down

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down :
            self.down_shift = lambda x : down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        """this class is a down shifted convolution layer, it takes an input x and applies a convolution operation on it

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right :
            self.right_shift = lambda x : right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                                stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x



from torch import nn
from torch.nn import functional as F
from torch.nn import init
class PolyakAveragedModel(nn.Module):
    def __init__(self, input_dim, num_filters, use_ema=False, ema_decay=0.999):
        super(PolyakAveragedModel, self).__init__()
        self.use_ema = use_ema
        self.hw = nn.Parameter(torch.empty(input_dim, 2 * num_filters)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        init.normal_(self.hw, mean=0, std=0.05)  # Initialize weights

        if self.use_ema:
            self.ema_decay = ema_decay
            self.ema_hw = torch.empty_like(self.hw).copy_(self.hw).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def update_ema(self):
        with torch.no_grad():
            self.ema_hw.mul_(self.ema_decay).add_(self.hw * (1 - self.ema_decay))

    def forward(self, h):
        hw = self.ema_hw if self.use_ema else self.hw
        return torch.matmul(h, hw)

class gated_resnet_plus(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=F.elu, skip_connection=0, input_dim=4):
        super(gated_resnet_plus, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)
        self.dropout = nn.Dropout2d(0.5)
        self.num_filters = num_filters
        
        sizes = {"u8":8, "d8":8, "u16":16, "d16":16, "u32":32, "d32":32}
        self.V_w_dict = {}
        self.V_b_dict = {}
        self.bias_w_dict = {}
        self.bias_b_dict = {}
        for size in sizes.keys():
            self.V_w_dict[size] = nn.Parameter(torch.empty(4, sizes[size])).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.V_b_dict[size] = nn.Parameter(torch.empty(4, sizes[size])).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Initialize parameters
            nn.init.xavier_uniform_(self.V_w_dict[size])
            nn.init.xavier_uniform_(self.V_b_dict[size])
        
        

        if skip_connection != 0:
            self.nin_skip = conv_op(2 * skip_connection * num_filters, num_filters)  # assuming conv_op is adaptable

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PKA = PolyakAveragedModel(input_dim, num_filters, use_ema=True).to(device)

        self.mode = "B"

    def forward(self, og_x, label, a=None, mode=None):

        # result tensor
        result_t = None
        
        if self.mode == "A":
            x = self.conv_input(self.nonlinearity(og_x))
            if a is not None:
                x += self.nin_skip(self.nonlinearity(a))
            x = self.nonlinearity(x)
            x = self.dropout(x)
            x = self.conv_out(x)
            h_weighted = self.PKA(label)  # Output size should be [batch_size, 2 * num_filters]

            # Ensure h_weighted is properly reshaped for broadcasting
            # Shape [batch_size, 2 * num_filters, 1, 1]
            h_weighted = h_weighted.view(x.size(0), 2 * self.num_filters, 1, 1)

            # Broadcasting here is implicit when adding tensors of shape [batch_size, 2 * num_filters, 1, 1] and [batch_size, 2 * num_filters, height, width]
            x += h_weighted  # Broadcasting works by aligning trailing dimensions
            if self.training:
                self.PKA.update_ema()

            a, b = x.chunk(2, dim=1)
            c3 = a * F.sigmoid(b)
            result_t = c3 + og_x
        
        if self.mode == "B":
            # mat mul with torch
            #load the correct weights and biases
            if mode == None:
                raise Exception("Mode not set")
            V_w = self.V_w_dict[mode]
            V_b = self.V_b_dict[mode]
            
            if label != None:
                b_w = torch.matmul(label, V_w)
                b_b = torch.matmul(label, V_b)

                b_w_shape = b_w.size()
                b_w = torch.reshape(b_w, (b_w_shape[0], 1, 1, b_w_shape[1]))
                b_b_shape = b_b.size()
                b_b = torch.reshape(b_b, (b_b_shape[0], 1, 1, b_b_shape[1]))

            x = self.conv_input(self.nonlinearity(og_x))
            if a is not None:
                x += self.nin_skip(self.nonlinearity(a))
            x = self.nonlinearity(x)
            x = self.dropout(x)
            x = self.conv_out(x)
            
            s_w, s_b = x.chunk(2, dim=1)
            result_t = torch.multiply(F.tanh(s_w + b_w), F.sigmoid(s_b + b_b))

        if self.mode == "C":
            # label is one hot of 4 class, add 4 additional channels, each channel is the same as the label
            # label shape is sample size by class count(4)
            
            # this expands the label to the same size as the input
            label = label.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32)

            # concatenate the label to the input
            x = torch.cat([og_x, label], dim=1)
            

            x = self.conv_input(self.nonlinearity(x))
            if a is not None:
                x += self.nin_skip(self.nonlinearity(a))
            x = self.nonlinearity(x)
            x = self.dropout(x)
            x = self.conv_out(x)
            h_weighted = self.PKA(label)  # Output size should be [batch_size, 2 * num_filters]

            # Ensure h_weighted is properly reshaped for broadcasting
            # Shape [batch_size, 2 * num_filters, 1, 1]
            h_weighted = h_weighted.view(x.size(0), 2 * self.num_filters, 1, 1)

            # Broadcasting here is implicit when adding tensors of shape [batch_size, 2 * num_filters, 1, 1] and [batch_size, 2 * num_filters, height, width]
            x += h_weighted  # Broadcasting works by aligning trailing dimensions
            if self.training:
                self.PKA.update_ema()

            a, b = x.chunk(2, dim=1)
            c3 = a * F.sigmoid(b)
            result_t = c3 + og_x

        
        return result_t
    

'''
skip connection parameter : 0 = no skip connection
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''
class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters) # cuz of concat elu

        if skip_connection != 0 :
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        # we apply dropout after the first layer and before the last layer
        # dropout chance is 0.5
        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)


    def forward(self, og_x, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None :
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)

        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * F.sigmoid(b)
        return og_x + c3
    