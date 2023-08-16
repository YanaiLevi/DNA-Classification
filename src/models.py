
### Authors: Tomer Gotesdyner and Yanai Levi ###

### Imports ###
import torch
import torch.nn.functional as F
import numpy
import torch.nn as nn
from torch.autograd import Variable




def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)  
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(1, 5), stride=(1, 1),
                              padding=(0, 3), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(1, 5), stride=(1, 1),
                              padding=(0, 3), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(1, 5), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x
    


class CNN10(nn.Module):
    def __init__(self,batch_size_val, batch_size_train, num_of_classes, device):
        super(CNN10, self).__init__() 
        factor = 4 
        self.device = device   
        self.batch_size_val = batch_size_val  
        self.batch_size_train = batch_size_train 
        self.bn0_val = nn.BatchNorm2d(batch_size_val)
        self.bn0_train = nn.BatchNorm2d(batch_size_train)
        self.conv_block1 = ConvBlock(1,64*factor)
        self.conv_block2 = ConvBlock(64*factor,128*factor)
        self.conv_block3 = ConvBlock(128*factor,256*factor)
        self.conv_block4 = ConvBlock(256*factor,512*factor)
    
        self.fc1 = nn.Linear(512*factor, 512*factor, bias=True)
        self.fc2 = nn.Linear(512*factor, 256*factor, bias=True)
        self.fc_audioset = nn.Linear(256*factor, num_of_classes, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0_train)
        init_bn(self.bn0_val)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
    
    def forward(self, x):
        x = x.transpose(0, 1)
        
        # check if training or validation and use the correct batch norm layer
        if self.training:
            if x.size(1) < self.batch_size_train:
                # padding the batch norm layer with zeros
                delta = self.batch_size_train - x.size(1)
                x = torch.cat((x, torch.zeros(x.size(0), delta, x.size(2), x.size(3)).to(self.device)), dim=1)
                x = self.bn0_train(x)
                x = x[:, :self.batch_size_train-delta, :, :]
            else:
                x = self.bn0_train(x)
        else:
            if x.size(1) < self.batch_size_val:
                # padding the batch norm layer with zeros
                delta = self.batch_size_val - x.size(1)
                x = torch.cat((x, torch.zeros(x.size(0), delta, x.size(2), x.size(3)).to(self.device)), dim=1)
                x = self.bn0_val(x)
                x = x[:, :self.batch_size_val-delta, :, :]
            else:
                x = self.bn0_val(x)
            
        x = x.transpose(0, 1)
        
        x = self.conv_block1(x, pool_size=(1, 5), pool_type='avg')
        
        # Performs only in training - drops out 0.2% of weights
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 5), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 5), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 5), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = F.relu_(self.fc1(x))
        x = F.relu_(self.fc2(x))
        embedding = x
        x = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = F.softmax(self.fc_audioset(x), dim=1, dtype=float)
        # clipwise_output = torch.sigmoid(self.fc_audioset(x)).float()
        
        return clipwise_output,embedding
        

    def predict(self, x, verbose = False):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob  and att
        # If verbose == False, only return global_prob

        result = []
        for i in range(0, len(x), self.batch_size_val):
            with torch.no_grad():
                input = Variable(x[i : i + self.batch_size_val]).cuda()
                output = self.forward(input)
                if not verbose: output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result[1] if verbose else result[0]
    
    
    
class CNN1DModel(nn.Module):
    def __init__(self, input_size, num_classes, network_size):
        super(CNN1DModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=network_size, kernel_size=5, padding=2),  # Adjust kernel size and padding
            nn.BatchNorm1d(network_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=network_size, out_channels=network_size*2, kernel_size=5, padding=2),  # Adjust kernel size and padding
            nn.BatchNorm1d(network_size*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=network_size*2, out_channels=network_size*4, kernel_size=5, padding=2),  # Adjust kernel size and padding
            nn.BatchNorm1d(network_size*4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=network_size*4, out_channels=network_size*8, kernel_size=5, padding=2),  # Adjust kernel size and padding
            nn.BatchNorm1d(network_size*8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=network_size*8, out_channels=network_size*16, kernel_size=5, padding=2),  # Adjust kernel size and padding
            nn.BatchNorm1d(network_size*16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # nn.Conv1d(in_channels=network_size*16, out_channels=network_size*16, kernel_size=5, padding=2),  # Adjust kernel size and padding
            # nn.BatchNorm1d(network_size*16),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            
            # nn.Conv1d(in_channels=network_size*16, out_channels=network_size*16, kernel_size=5, padding=2),  # Adjust kernel size and padding
            # nn.BatchNorm1d(network_size*16),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),
            
            # nn.Conv1d(in_channels=network_size*16, out_channels=network_size*16, kernel_size=5, padding=2),  # Adjust kernel size and padding
            # nn.BatchNorm1d(network_size*16),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2)
        )
        
        # Calculate the output size after convolutional layers
        self.num_conv_output = input_size // (2**5)  # Dividing by 2^5 due to 5 max pooling layers
        
        self.fc_layers = nn.Sequential(
            nn.Linear(network_size*16 * self.num_conv_output, network_size*8),
            # nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(network_size*8, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x, 0