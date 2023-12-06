import torch
from torchray.attribution.gradient import gradient
import numpy as np

def gradbox(net,a,l, length=16, p=0.5,rescle = True, arng=[8,18], sal=False):
    '''
    a : batch of images
    p: probability of gradient happening
    '''
    bsize = a.shape[0]
    if length ==0:
        lengt=torch.randint(arng[0], arng[1], (bsize,)).cuda()
    else:
        lengt = length
    net.eval()
    saliency = gradient(net, a, l)
    n = saliency.view(bsize, -1)
    _, ll = torch.max(n,1)
    x, y = torch.div(ll, 32, rounding_mode='floor'), torch.remainder(ll,32)
    y1 = torch.clamp(y - (torch.div(lengt, 2, rounding_mode='floor')), 0, 32)
    y2 = torch.clamp(y + (torch.div(lengt, 2, rounding_mode='floor')), 0, 32)
    x1 = torch.clamp(x - (torch.div(lengt, 2, rounding_mode='floor')), 0, 32)
    x2 = torch.clamp(x + (torch.div(lengt, 2, rounding_mode='floor')), 0, 32)
    new = torch.empty((1,32,32)).cuda()
    if sal:
        for ii in range(bsize):
            mask = torch.zeros((1, 32,32)).cuda()
            mask[0, x1[ii]: x2[ii], y1[ii]: y2[ii]] = 1.
            new = torch.cat((new,mask),dim = 0)
    else:
        for ii in range(bsize):
            mask = torch.ones((1, 32,32)).cuda()
            mask[0, x1[ii]: x2[ii], y1[ii]: y2[ii]] = 0.
            new = torch.cat((new,mask),dim = 0)
    
    masks = new[1:,:,:].unsqueeze(1).expand_as(a)
    AAA = masks.expand_as(a)
    fo = a*AAA
    if rescle:
        fo = fo * (1/(lengt**2/32**2))
    rnd = ((torch.rand(size=(bsize,1,1,1)) < p).int()).cuda()
    foo = rnd*fo
    aa = ((1-rnd)*a) + foo
    net.train()
    return aa