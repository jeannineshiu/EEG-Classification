from dataloader import read_bci_data
import argparse
import torch
import torch.nn as nn
import torch.utils.data as utils
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
import numpy as np

Activations = nn.ModuleDict([
    ['relu', nn.ReLU()],
    ['lrelu', nn.LeakyReLU()],
    ['elu', nn.ELU()],
])

class DeepConvNet(nn.Module):
    def __init__(self, act):
        super(DeepConvNet, self).__init__()
        p = 0.8
        self.block1 = nn.Sequential()
        # [B, 1, 2, 750] -> [B, 25, 2, 750]
        self.block1.add_module('conv1', nn.Conv2d(1, 25, (1, 5), bias=False))
        # [B, 25, 2, 750] -> [B, 25, 1, 750]
        self.block1.add_module('conv2', nn.Conv2d(25, 25, (2, 1), bias=False))
        self.block1.add_module('norm1', nn.BatchNorm2d(25,momentum=None,track_running_stats=False))
        self.block1.add_module('act1', Activations[act])
        # [B, 25, 2, 750] -> [B, 25, 1, 373]
        self.block1.add_module('pool1', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block1.add_module('drop1', nn.Dropout(p))

        self.block2 = nn.Sequential()
        # [B, 25, 1, 373] -> [B, 50, 1, 373]
        self.block2.add_module('conv3', nn.Conv2d(25, 50, (1, 5), bias=False))
        self.block2.add_module('norm2', nn.BatchNorm2d(50,momentum=None,track_running_stats=False))
        self.block2.add_module('act2', Activations[act])
        # [B, 50, 1, 373] -> [B, 50, 1, 184]
        self.block2.add_module('pool2', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block2.add_module('drop2', nn.Dropout(p))
        
        self.block3 = nn.Sequential()
        # [B, 50, 1, 184] -> [B, 100, 1, 184]
        self.block3.add_module('conv4', nn.Conv2d(50, 100, (1, 5), bias=False))
        self.block3.add_module('norm3', nn.BatchNorm2d(100,momentum=None,track_running_stats=False))
        self.block3.add_module('act3', Activations[act])
        # [B, 100, 1, 184] -> [B, 100, 1, 90]
        self.block3.add_module('pool3', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block3.add_module('drop3', nn.Dropout(p))

        self.block4 = nn.Sequential()
        # [B, 100, 1, 90] -> [B, 200, 1, 90]
        self.block4.add_module('conv5', nn.Conv2d(100, 200, (1, 5), bias=False))
        self.block4.add_module('norm4', nn.BatchNorm2d(200,momentum=None,track_running_stats=False))
        self.block4.add_module('act4', Activations[act])
        # [B, 200, 1, 90] -> [B, 200, 1, 43]
        self.block4.add_module('pool4', nn.MaxPool2d((1, 2), stride=(1, 2)))
        self.block4.add_module('drop4', nn.Dropout(p))

        self.classify = nn.Sequential(
            nn.Linear(200*43, 2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        res = x.view(x.size(0), -1)     # [B, 200, 1, 43] -> [B, 200 * 43]
        out = self.classify(res)
        return out


# class DeepConvNet(nn.Module):
#   def __init__(self, act):
#     super().__init__()
#     p = 0.8
#     bn_args = {
#         #'eps': 1e-5,
#         #'momentum': 0.1,
#         #'eps': 1e-2,
#         #'momentum': 0.99,
#         #'track_running_stats': True,
#         'momentum': None,
#         'track_running_stats': False,
#     }
#     self.flat_dim = 8600
#     conv_kernel = (1, 5)
#     pool_kernel, pool_stride = (1, 2), (1, 2)
#     #self.flat_dim = 800
#     #conv_kernel = (1, 11)
#     #pool_kernel, pool_stride = (1, 3), (1, 3)
#     self.conv0 = nn.Sequential(
#         nn.Conv2d(1, 25, kernel_size=conv_kernel),
#         nn.Conv2d(25, 25, kernel_size=(2, 1)),
#         nn.BatchNorm2d(25, **bn_args),
#         Activations[act],
#         nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
#         nn.Dropout(p),
#     )
#     channels = [25, 50, 100, 200]
#     self.convs = nn.ModuleList([
#         nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel),
#             nn.BatchNorm2d(out_channels, **bn_args),
#             Activations[act],
#             nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
#             nn.Dropout(p),
#         ) for in_channels, out_channels in zip(channels[:-1], channels[1:])
#     ])
#     self.classify = nn.Sequential(
#         nn.Linear(in_features=self.flat_dim, out_features=2, bias=True))

#   def forward(self, x):
#     x = self.conv0(x)
#     for conv in self.convs:
#       x = conv(x)
#     x = x.view(-1, self.flat_dim)
#     x = self.classify(x)
#     return x


class EEGNet(nn.Module):
  def __init__(self, act, ks=51, p=0.25, flat_dim=736, F1=16, D=2, F2=32):
    super().__init__()
    self.flat_dim = flat_dim
    self.firstConv = nn.Sequential(
        nn.Conv2d(1, F1, kernel_size=(1, ks), padding=(0,25), bias=False),
        nn.BatchNorm2d(F1,momentum=None,track_running_stats=False),
    )
    self.depthwiseConv = nn.Sequential(
        nn.Conv2d(F1, D * F1, kernel_size=(2, 1), groups=F1, bias=False),
        nn.BatchNorm2d(D * F1,momentum=None,track_running_stats=False),
        Activations[act],
        nn.AvgPool2d(kernel_size=(1, 4),stride =(1, 4),padding = 0),
        nn.Dropout(p),
    )
    self.separableConv = nn.Sequential(
        nn.Conv2d(D * F1, F2, kernel_size=(1, 15), padding=(0,7), bias=False),
        nn.BatchNorm2d(F2,momentum=None,track_running_stats=False),
        Activations[act],
        nn.AvgPool2d(kernel_size=(1, 8),stride=(1, 8),padding = 0),
        nn.Dropout(p),
    )
    self.classify = nn.Sequential(
        nn.Linear(in_features=self.flat_dim, out_features=2, bias=True))

  def forward(self, x):
    x = self.firstConv(x)
    x = self.depthwiseConv(x)
    x = self.separableConv(x)
    x = x.view(-1, self.flat_dim)
    x = self.classify(x)
    return x


def get_dataloader(batch_size, device):
  def get_loader(data, label):
    data = torch.stack([torch.Tensor(i) for i in data]).to(device)
    label = torch.LongTensor(label).to(device)
    dataset = utils.TensorDataset(data, label)
    loader = utils.DataLoader(dataset, batch_size)
    return loader
  # torch_dataset = Data.TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32)))
  # train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch, shuffle=True)
  
  train_data, train_label, test_data, test_label = read_bci_data()
  train_loader = get_loader(train_data, train_label)
  test_loader = get_loader(test_data, test_label)
  return train_loader, test_loader

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(args):
  arg = [args.activation]
  if args.net == 'EEGNet':
    arg += [args.kernel_size, args.drop_prob, args.flat_dim, args.F1, args.D]
  net = Nets[args.net](*arg).to(args.device)
  # print(net)
  criterion = nn.CrossEntropyLoss()
  # opt = torch.optim.RMSprop
  optim = torch.optim.Adam
  optimizer = optim(
      net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  train_loader, test_loader = get_dataloader(args.batch_size, args.device)
  softmax = nn.Softmax(dim=1)

  max_acc = 0
  file_name = '{}_lr{}_ep{}'.format(args.net, args.lr, args.epochs)
  acc_dict = {}
  acc_dict['train_{}'.format(args.activation)] = []
  acc_dict['test_{}'.format(args.activation)] = []

  for epoch in range(args.epochs):
    correct, total = 0, 0
    for i, (inputs, labels) in enumerate(train_loader, start=1):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # batch train accuracy
      with torch.no_grad():
        predicted = torch.argmax(softmax(outputs), dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # train accuracy
    train_acc = correct / total

    # test accuracy
    correct, total = 0, 0
    net.eval()
    with torch.no_grad():
      for inputs, labels in test_loader:
        outputs = net(inputs)
        predicted = torch.argmax(softmax(outputs), dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_acc = correct / total
    if test_acc > max_acc:
       max_acc = test_acc
    acc_dict['train_{}'.format(args.activation)].append(train_acc)
    acc_dict['test_{}'.format(args.activation)].append(test_acc)
    print('[{:4d}] train acc: {:.2%} test acc: {:.2%}'.format(
        epoch + 1, train_acc, test_acc))
    
  print('max_acc: {}'.format(max_acc))
    
  with open(file_name + '.json', 'w') as f:
    json.dump({
        'x': list(range(args.epochs)),
        'y_dict': acc_dict,
        'title': args.net,
    }, f, cls=NumpyEncoder)

Nets = {
    'DeepConvNet': DeepConvNet,
    'EEGNet': EEGNet,
}

# python3 main.py --net EEGNet -lr 0.001 --weight_decay 0.06 --epochs 1200 --batch_size 128 --activation elu
# python3 main.py --net EEGNet -lr 0.001 --weight_decay 0.06 --epochs 1000 --batch_size 64 --activation elu

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # network
  parser.add_argument('-n', '--net', default='EEGNet', choices=Nets.keys())
  parser.add_argument(
      '-act', '--activation', default='relu', choices=Activations.keys())
  # EEGNet hyper-parameter
  parser.add_argument('-ks', '--kernel_size', default=51, type=int)
  parser.add_argument('-p', '--drop_prob', default=0.25, type=float)
  parser.add_argument('-F1', '--F1', default=16, type=int)
  parser.add_argument('-D', '--D', default=2, type=int)
  parser.add_argument('-flat_dim', '--flat_dim', default=736, type=int)
  # training
  parser.add_argument('-d', '--device', default='cuda')
  parser.add_argument('-bs', '--batch_size', default=64, type=int)
  parser.add_argument('-e', '--epochs', default=600, type=int)
  parser.add_argument('-lr', '--lr', default=0.001, type=float)
  parser.add_argument('-wd', '--weight_decay', default=0, type=float)
  #parser.add_argument("-load", "--load", help="your pkl file path", type=str, default='')

  args = parser.parse_args()
  main(args)