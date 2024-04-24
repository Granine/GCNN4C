import torch.nn as nn
from layers import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, 
                                                         down_shifted_conv2d,
                                        resnet_nonlinearity, 
                                        skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, label):
        """ forward pass of PixelCNNLayer_up, it takes u and ul as input and apply the gated resnet

        Args:
            u (_type_): _description_
            ul (_type_): _description_

        Returns:
            _type_: _description_
        """
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, label=label, mode = 'u' + str(u.size()[2]))
            ul = self.ul_stream[i](ul, a=u, label=label, mode = 'u' + str(ul.size()[2]))
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list, label):
        for i in range(self.nr_resnet):
            # bad practice, you should not be poping across class
            u_poped = u_list.pop()
            u  = self.u_stream[i](u, a=u_poped, label=label, mode='d' + str(u_poped.size()[2]))
            ul_poped = ul_list.pop()
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_poped), 1), label=label, mode='d' + str(ul_poped.size()[2]))

        return u, ul
    
class ConditionalNorm(nn.Module):
    def __init__(self, num_features, num_classes, norm='weight_norm'):
        super(ConditionalNorm, self).__init__()
        self.num_features = num_features
        
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialize gamma at 1
        self.embed.weight.data[:, num_features:].zero_()  # Initialize beta at 0
        if norm == 'weight_norm':
            self.conv = wn(nn.Conv2d(num_features, num_features, 3, padding=1))
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_features, affine=False)

    def forward(self, x, y):
        """ forward pass of ConditionalNorm, it takes x and y as input and apply the normalization
        
        Args:
            x: input feature map
            y: input class label, one hot encoded
        """
        # check if input is one-hot encoded, if so y = torch.argmax(y_onehot, dim=1) 
        if len(y.size()) == 2:
            y = torch.argmax(y, dim=1)

        if self.norm == 'weight_norm':
            out = self.bn(x)
        elif self.norm == 'batch_norm':
            out = self.conv(x)

        # need gamma beta to control the impact of labels
        gamma, beta = self.embed(y).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return gamma * out + beta


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3,
                    num_classes=4, embedding_dim=16
                    ):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))

        self.num_classes = num_classes
        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2

        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim)
        self.onehot_linear = nn.Linear(num_classes, nr_filters)

        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None


    def forward(self, x, sample=False, class_label:torch.tensor=None):

        # Get class embeddings
        # if class_label is not None:
        #     class_embeddings = self.class_embedding(class_label)
        #     x = torch.cat([(x, class_embeddings.unsqueeze(0))], dim=1)
        # else :
        #     # exception
        #     raise Exception("Class label is not provided")
        
        # similar as done in the tf repo :
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        x = x if sample else torch.cat((x, self.init_padding), 1)

        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            # mode = f/b + size (eg: 32, 16, 8) ie "f8"
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], label=class_label)
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, label=class_label)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        # output is [batch, nr_logistic_mix * nr_channels, height, width]
        # where nr_logistic_mix is 10 in the experiments
        # and nr_channels is 3 (RGB)
        return x_out
    
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if 'models' not in os.listdir():
            os.mkdir('models')
        torch.save(self.state_dict(), 'models/conditional_pixelcnn.pth')
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
    
    