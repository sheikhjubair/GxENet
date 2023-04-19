#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:





# In[2]:


class fc_net(nn.Module):
    def __init__(self, num_env, num_geno, reduce_dim=2666):
        super(fc_net, self).__init__()
        
        self.reduce_dim = reduce_dim
        self.num_geno = num_geno
        self.num_env = num_env

        self.fc1 = nn.Linear(num_geno, reduce_dim)
        self.batchnorm1 = nn.BatchNorm1d(reduce_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        
        
        self.fc2 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm2 = nn.BatchNorm1d(reduce_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm3 = nn.BatchNorm1d(reduce_dim)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm4 = nn.BatchNorm1d(reduce_dim)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm5 = nn.BatchNorm1d(reduce_dim)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.5)

        self.fc6 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm6 = nn.BatchNorm1d(reduce_dim)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm7 = nn.BatchNorm1d(reduce_dim)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(p=0.5)

        self.fc8 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm8 = nn.BatchNorm1d(reduce_dim)
        self.relu8 = nn.ReLU()
        self.drop8 = nn.Dropout(p=0.5)

        self.fc9 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm9 = nn.BatchNorm1d(reduce_dim)
        self.relu9 = nn.ReLU()
        self.drop9 = nn.Dropout(p=0.5)

        self.fc10 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm10 = nn.BatchNorm1d(reduce_dim)
        self.relu10 = nn.ReLU()
        self.drop10 = nn.Dropout(p=0.5)

        self.fc11 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm11 = nn.BatchNorm1d(reduce_dim)
        self.relu11 = nn.ReLU()
        self.drop11 = nn.Dropout(p=0.5)

        self.fc12 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm12 = nn.BatchNorm1d(reduce_dim)
        self.relu12 = nn.ReLU()
        self.drop12 = nn.Dropout(p=0.5)
        
        self.linears = nn.ModuleList([nn.Linear(num_env + 1, 54) for i in range(reduce_dim)])
        
        
        
        self.fc13 = nn.Linear(54, 54)
        self.batchnorm13 = nn.BatchNorm1d(750)
        self.relu13 = nn.ReLU()
        # self.drop13 = nn.Dropout(p=0.3)
        
        
        self.fc14 = nn.Linear(54, 54)
        self.batchnorm14 = nn.BatchNorm1d(750)
        self.relu14 = nn.ReLU()
        # self.drop14 = nn.Dropout(p=0.3)
        
        self.fc15 = nn.Linear(54, 5)
        self.batchnorm15 = nn.BatchNorm1d(750)
        self.relu15 = nn.ReLU()
        # self.drop15 = nn.Dropout(p=0.3)
        
        self.fc16 = nn.Linear(3750, 750)
        self.relu16 = nn.ReLU()
        
        self.fc17 = nn.Linear(750, 750)
        self.relu17 = nn.ReLU()
        
        self.fc18 = nn.Linear(750, 750)
        self.relu18 = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        env = x[:, self.num_geno:  ]
        x = x[:, : self.num_geno]
        x = x.view(batch_size, -1)
        
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        d = x
        # x = self.maxpool1(x)

        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = d + x
        d = x

        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        
        x = self.fc5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.drop5(x)

        x = d + x
        d = x
        
        x = self.fc6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = d + x
        d = x
        
        x = self.fc8(x)
        x = self.batchnorm8(x)
        x = self.relu8(x)
        x = self.drop8(x)

        x = self.fc9(x)
        x = self.batchnorm9(x)
        x = self.relu9(x)
        x = self.drop9(x)

        x = d + x
        d = x
        
        x = self.fc10(x)
        x = self.batchnorm10(x)
        x = self.relu10(x)
        x = self.drop10(x)

        x = self.fc11(x)
        x = self.batchnorm11(x)
        x = self.relu11(x)
        x = self.drop11(x)

        x = d + x
        d = x
        
        x = self.fc12(x)
        x = self.batchnorm12(x)
        x = self.relu12(x)
        x = self.drop12(x)
        # x = d + x
        # d = x

        x = x.view(batch_size, -1)

        d = torch.zeros([batch_size, self.reduce_dim, 54]).to(device)
        for i, layer in enumerate(self.linears):
          geno = x[:, i]
          geno = torch.unsqueeze(geno, 1)
          z = torch.cat((env, geno), dim=1)
       
          out = layer(z)
          d[:, i, :] = out
        
        # x = self.relu4(d)
        
        x = d
        
        x = self.fc13(x)
        x = self.batchnorm13(x)
        x = self.relu13(x)
        # x = self.drop13(x)
        
        x = self.fc14(x)
        x = self.batchnorm14(x)
        x = self.relu14(x)
        # x = self.drop14(x)
        
        # x = d + x
        # d = x
        
        x = self.fc15(x)
        x = self.batchnorm15(x)
        x = self.relu15(x)
        # x = self.drop15(x)
        
        x= x.view(batch_size, -1)
        
        x = self.fc16(x)
        x = self.relu16(x)
        
        x = self.fc17(x)
        x = self.relu17(x)
        
        x = self.fc18(x)
        x = self.relu18(x)

        
        return x


# In[3]:


class fc_net_v2(nn.Module):
    def __init__(self, num_env, num_geno, reduce_dim=2666):
        super(fc_net_v2, self).__init__()
        
        self.reduce_dim = reduce_dim
        self.num_geno = num_geno
        self.num_env = num_env

        self.fc1 = nn.Linear(num_geno, reduce_dim)
        self.batchnorm1 = nn.BatchNorm1d(reduce_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.5)
        
        
        self.fc2 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm2 = nn.BatchNorm1d(reduce_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm3 = nn.BatchNorm1d(reduce_dim)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm4 = nn.BatchNorm1d(reduce_dim)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.5)

        self.fc5 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm5 = nn.BatchNorm1d(reduce_dim)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.5)

        self.fc6 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm6 = nn.BatchNorm1d(reduce_dim)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(reduce_dim, reduce_dim)
        self.batchnorm7 = nn.BatchNorm1d(reduce_dim)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout(p=0.5)

#         self.fc8 = nn.Linear(reduce_dim, reduce_dim)
#         self.batchnorm8 = nn.BatchNorm1d(reduce_dim)
#         self.relu8 = nn.ReLU()
#         self.drop8 = nn.Dropout(p=0.5)

#         self.fc9 = nn.Linear(reduce_dim, reduce_dim)
#         self.batchnorm9 = nn.BatchNorm1d(reduce_dim)
#         self.relu9 = nn.ReLU()
#         self.drop9 = nn.Dropout(p=0.5)

#         self.fc10 = nn.Linear(reduce_dim, reduce_dim)
#         self.batchnorm10 = nn.BatchNorm1d(reduce_dim)
#         self.relu10 = nn.ReLU()
#         self.drop10 = nn.Dropout(p=0.5)

#         self.fc11 = nn.Linear(reduce_dim, reduce_dim)
#         self.batchnorm11 = nn.BatchNorm1d(reduce_dim)
#         self.relu11 = nn.ReLU()
#         self.drop11 = nn.Dropout(p=0.5)

#         self.fc12 = nn.Linear(reduce_dim, reduce_dim)
#         self.batchnorm12 = nn.BatchNorm1d(reduce_dim)
#         self.relu12 = nn.ReLU()
#         self.drop12 = nn.Dropout(p=0.5)
        
        
        
        
        
        self.fc13 = nn.Linear(reduce_dim + num_env, 750)
        self.batchnorm13 = nn.BatchNorm1d(750)
        self.relu13 = nn.ReLU()
        # self.drop13 = nn.Dropout(p=0.3)
        
        
        self.fc14 = nn.Linear(750, 750)
        self.batchnorm14 = nn.BatchNorm1d(750)
        self.relu14 = nn.ReLU()
        # self.drop14 = nn.Dropout(p=0.3)
        
#         self.fc15 = nn.Linear(750, 5)
#         self.batchnorm15 = nn.BatchNorm1d(750)
#         self.relu15 = nn.ReLU()
#         # self.drop15 = nn.Dropout(p=0.3)
        
#         self.fc16 = nn.Linear(3750, 750)
#         self.relu16 = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        env = x[:, self.num_geno:  ]
        x = x[:, : self.num_geno]
        x = x.view(batch_size, -1)
        
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        # d = x
        # x = self.maxpool1(x)

        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        # x = d + x
        # d = x

        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        
        x = self.fc5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.drop5(x)

        # x = d + x
        # d = x
        
        x = self.fc6(x)
        x = self.batchnorm6(x)
        x = self.relu6(x)
        x = self.drop6(x)
        
        x = self.fc7(x)
        x = self.batchnorm7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        # x = d + x
        # d = x
        
#         x = self.fc8(x)
#         x = self.batchnorm8(x)
#         x = self.relu8(x)
#         x = self.drop8(x)

#         x = self.fc9(x)
#         x = self.batchnorm9(x)
#         x = self.relu9(x)
#         x = self.drop9(x)

#         x = d + x
#         d = x
        
#         x = self.fc10(x)
#         x = self.batchnorm10(x)
#         x = self.relu10(x)
#         x = self.drop10(x)

#         x = self.fc11(x)
#         x = self.batchnorm11(x)
#         x = self.relu11(x)
#         x = self.drop11(x)

#         x = d + x
#         d = x
        
#         x = self.fc12(x)
#         x = self.batchnorm12(x)
#         x = self.relu12(x)
#         x = self.drop12(x)
        # x = d + x
        # d = x

        x = torch.cat((x, env), dim=1)

       
        
        # x = d
        
        x = self.fc13(x)
        x = self.batchnorm13(x)
        x = self.relu13(x)
        # x = self.drop13(x)
        
        x = self.fc14(x)
        x = self.batchnorm14(x)
        x = self.relu14(x)
        # x = self.drop14(x)
        
        # x = d + x
        # d = x
        
#         x = self.fc15(x)
#         x = self.batchnorm15(x)
#         x = self.relu15(x)
#         # x = self.drop15(x)
        
#         x= x.view(batch_size, -1)
        
#         x = self.fc16(x)
#         x = self.relu16(x)

        
        return x


# In[4]:


class fc_env_middle(nn.Module):
    def __init__(self, num_env, num_geno, reduce_dim=2666, output_dim=1):
        super(fc_env_middle, self).__init__()

        self.reduce_dim = reduce_dim
        self.num_geno = num_geno
        self.num_env = num_env

        self.fc_net = fc_net(num_env, num_geno, reduce_dim)
        
        
        self.regress = nn.Linear(750, 1)
        
    def forward(self, x):
        x = self.fc_net(x)
        x = self.regress(x)
        
        return x


# In[5]:


class fc_avg_net(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(fc_avg_net, self).__init__()
        
        self.representation = None
        
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.relu1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.LeakyReLU(0.1)
        self.drop3 = nn.Dropout(p=0.5)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = nn.LeakyReLU(0.1)
        self.drop4 = nn.Dropout(p=0.5)
        
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu5 = nn.LeakyReLU(0.1)
        self.drop5 = nn.Dropout(p=0.5)
        
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu5 = nn.LeakyReLU(0.1)
        self.drop5 = nn.Dropout(p=0.5)
        
        self.fc6 = nn.Linear(hidden_dim, 666)
        self.relu6 = nn.LeakyReLU(0.1)
        self.drop6 = nn.Dropout(p=0.4)
        
        self.fc7 = nn.Linear(666, 444)
        self.relu7 = nn.LeakyReLU(0.1)
        self.drop7 = nn.Dropout(p=0.4)
        
        self.fc8 = nn.Linear(444, 296)
        self.relu8 = nn.LeakyReLU(0.1)
        self.drop8 = nn.Dropout(p=0.2)
        
        self.fc9 = nn.Linear(296, 1)
        
        
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop3(out)
        
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.drop4(out)
        
        out = self.fc5(out)
        out = self.relu5(out)
        out = self.drop5(out)
        
        out = self.fc6(out)
        out = self.relu6(out)
        out = self.drop6(out)
        
        out = self.fc7(out)
        out = self.relu7(out)
        out = self.drop7(out)
        
        out = self.fc8(out)
        out = self.relu8(out)
        representation = out
        out = self.drop8(out)
        
        out = self.fc9(out)
        
        
        return out, representation
        


# In[6]:


class conv_avg_net(nn.Module):
    def __init__(self):
        super(conv_avg_net, self).__init__()
        
        self.representation = None
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,3), padding=(0,1))
        self.relu1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(p=0.6)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same')
        self.relu2 = nn.LeakyReLU(0.1)
        self.drop2 = nn.Dropout(p=0.6)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same')
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.6)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(4,1), stride=(4,1))
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,3), padding=(0,1))
        self.relu4 = nn.LeakyReLU(0.1)
        self.drop4 = nn.Dropout(p=0.5)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding='same')
        self.relu5 = nn.LeakyReLU(0.1)
        self.drop5 = nn.Dropout(p=0.5)
        
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding='same')
        self.relu6 = nn.LeakyReLU(0.1)
        self.drop6 = nn.Dropout(p=0.5)
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=(4,1), stride=(4,1))
        
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(4,3), padding=(0,1))
        self.relu7 = nn.LeakyReLU(0.1)
        self.drop7 = nn.Dropout(p=0.5)
        
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same')
        self.relu8 = nnn.LeakyReLU(0.1)
        self.drop8 = nn.Dropout(p=0.5)
        
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same')
        self.relu9 = nn.LeakyReLU(0.1)
        self.drop9 = nn.Dropout(p=0.5)
        
        self.maxpool3 = nn.MaxPool2d(kernel_size=(4,1), stride=(4,1))
        
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same')
        self.relu10 = nn.LeakyReLU(0.1)
        self.drop10 = nn.Dropout(p=0.5)
        
        self.conv11 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same')
        self.relu11 = nn.LeakyReLU(0.1)
        self.drop11 = nn.Dropout(p=0.5)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=(4,1), stride=(4,1))
        
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding='same')
        self.relu12 = nn.LeakyReLU(0.1)
        self.drop12 = nn.Dropout(p=0.5)
        
        self.conv13 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding='same')
        self.relu13 = nn.LeakyReLU(0.1)
        self.drop13 = nn.Dropout(p=0.5)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=10, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.fc1 = nn.Linear(in_features=1590, out_features= 785)
        self.relu14 = nn.LeakyReLU(0.1)
        self.drop14 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(in_features=785, out_features= 512)
        self.relu15 = nn.LeakyReLU(0.1)
        self.drop15 = nn.Dropout(p=0.6)
        
        self.fc3 = nn.Linear(in_features=512, out_features= 128)
        self.relu16 = nn.LeakyReLU(0.1)
        self.drop16 = nn.Dropout(p=0.4)
        
        self.regress = nn.Linear(in_features=128, out_features=1)
               
        
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], -1)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.drop3(out)
        
        out = self.maxpool1(out)
        
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.drop4(out)
        
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.drop5(out)
        
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.drop6(out)
        
        out = self.maxpool2(out)
        
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.drop7(out)
        
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.drop8(out)
        
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.drop9(out)
        
        out = self.maxpool3(out)
        
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.drop10(out)
        
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.drop11(out)
        
        out = self.maxpool4(out)
        
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.drop12(out)
        
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.drop13(out)
        
        # out = out.reshape(out.shape[0], -1, out.shape[-1])
        # out = self.transformer_encoder(out)
        
        out = out.reshape(out.shape[0],-1)
        
        out = self.fc1(out)
        out = self.relu14(out)
        out = self.drop14(out) 
        
        out = self.fc2(out)
        representation = out
        out = self.relu15(out)
        # out = self.drop15(out)
        
        out = self.fc3(out)
        representation = out
        out = self.relu16(out)
        out = self.drop16(out)
        
        out = self.regress(out)
        
        return out, representation
        


# In[ ]:


class fc_avg_net_over_geno(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(fc_avg_net_over_geno, self).__init__()
        
        self.representation = None
        
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.25)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.25)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.25)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.25)
        
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.25)
        
        self.fc5 = nn.Linear(hidden_dim, 1)
#         self.relu5 = nn.ReLU()
#         self.drop5 = nn.Dropout(p=0.5)
        
#         self.fc6 = nn.Linear(hidden_dim, 36)
#         self.relu6 = nn.ReLU()
#         self.drop6 = nn.Dropout(p=0.25)
        
#         self.fc7 = nn.Linear(36, 1)
        
        
        
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop3(out)
        
        out = self.fc4(out)
        
        out = self.relu4(out)
        representation = out
        out = self.drop4(out)
        
        
        
        # print(representation)
        out = self.fc5(out)
#         out = self.relu5(out)
#         out = self.drop5(out)
        
#         out = self.fc6(out)
#         out = self.relu6(out)
#         out = self.drop6(out)
#         representation = out
        
#         out = self.fc7(out)
        
        
        
        return out, representation
        


# In[ ]:


class final_net(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(final_net, self).__init__()
        
        self.representation = None
        
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.25)
        self.softmax1 = nn.Softmax(dim=1)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.25)
        self.softmax2 = nn.Softmax(dim=1)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.25)
        self.softmax3 = nn.Softmax(dim=1)
        
        self.fc4 = nn.Linear(hidden_dim, 1)
#         self.relu4 = nn.ReLU()
        
#         self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.drop1(out)
        # x = self.softmax1(out)
        # out = out * x
        
        out = self.fc2(out)
        out = self.relu2(out)
        # out = self.drop2(out)
        # out = self.drop2(out)
        # x = self.softmax2(out)
        # out = x * out
        
        out = self.fc3(out)
        out = self.relu3(out)
        # out = self.drop3(out)
        # out = self.drop3(out)
        # x = self.softmax3(out)
        # out = out * x
        
        # out = self.fc4(out)
        # out = self.relu4(out)
        
        out = self.fc4(out)    
        
        return out, None
        

