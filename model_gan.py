
import torch.nn as nn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

immean = 0.9664114577640158
imstd = 0.0858381272736797

model_gan = nn.Sequential( # Sequential,
	nn.Conv2d(1,48,(5, 5),(2, 2),(2, 2)),
	nn.ReLU(),
	nn.Conv2d(48,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(1024,512,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(512,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(256,256,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(128,128,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,48,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ConvTranspose2d(48,48,(4, 4),(2, 2),(1, 1),(0, 0)),
	nn.ReLU(),
	nn.Conv2d(48,24,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(24,1,(3, 3),(1, 1),(1, 1)),
	nn.Sigmoid(),
)