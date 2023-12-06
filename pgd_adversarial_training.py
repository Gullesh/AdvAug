import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from models import *
import argparse

from util.gradient import gradbox


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
data_options = ['cifar10', 'cifar100','imagenet']
model_options = ['resnet', 'resnet50','densenet', 'wrn','mobile','dla', 'resnet50']
parser.add_argument('--data', default='cifar10',  choices=data_options)
'''

# Model and Training parameters
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--data', default='cifar10',  choices=data_options)
# Augmentations and their parameters

parser.add_argument('--nworkers', type=int, default=2,
                    help='number of workers for trainloader')
'''
parser.add_argument('--model', default='resnet',  choices=model_options)

parser.add_argument('--re', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--gbox', action='store_true', default=False,
                    help='apply GradCAM')
parser.add_argument('--gboxadv', action='store_true', default=False,
                    help='apply GradCAM')
parser.add_argument('--sal', action='store_true', default=False,
                    help='if True only keep the salient part')
parser.add_argument('--rescle', action='store_true', default=False,
                    help='scaling random erasing')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--pexp', type=float, default=0.25, help='chance of explainablity augmentation happening')
parser.add_argument('--stepoch', type=int, default=0,
                   help='starting explain augmentation epoch')
parser.add_argument('--a', nargs="+", type=int ,help= 'the range of image areas(%) to be removed')
args = parser.parse_args()
print(args)

print(args.a)
learning_rate = 0.1
epsilon = 0.0314
k = 7
alpha = 0.00784
file_name = 'pgd_adversarial_training'

device = 'cuda' if torch.cuda.is_available() else 'cpu'




if args.data=='cifar10':
    n_class = 10
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
    if args.re:
        transform_train.transforms.append(transforms.RandomErasing(p=0.5,rescle=args.rescle))
    if args.cutout:
        transform_train.transforms.append(Cutout(p=args.pexp, rescle=args.rescle))
    
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=4)


elif args.data == 'cifar100':
    n_class = 100
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
    if args.re:
        transform_train.transforms.append(transforms.RandomErasing(p=args.pexp))
    if args.cutout:
        transform_train.transforms.append(Cutout(p=args.pexp))
    
    transform_test = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=4)




class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

def attack(x, y, model, adversary):
    model_copied = copy.deepcopy(model)
    model_copied.eval()
    adversary.model = model_copied
    adv = adversary.perturb(x, y)
    return adv



# Model
print('==> Building model..')
if args.model == 'resnet':
    net = ResNet18(num_classes=n_class)
elif args.model == 'wrn':
    net = wrn(num_classes=n_class)
elif args.model == 'mobile':
    net = MobileNetV2(num_classes=n_class)

elif args.model == 'dla':
    net = DLA(num_classes=n_class)
elif args.model == 'densenet':
    net = DenseNet121(num_classes=n_class)
elif args.model == 'resnet50':
    net = ResNet50(num_classes=n_class)
else:
    print('Error: please choose a valid model')


net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

def train(epoch):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if (args.gbox and epoch>=args.stepoch):
            inputs = gradbox(net,inputs,targets,p=args.pexp, length = args.length,rescle=args.rescle, arng=args.a,sal=args.sal)
        adv = adversary.perturb(inputs, targets)
        if (args.gboxadv and epoch>=args.stepoch):
            adv = gradbox(net,adv,targets,p=args.pexp, length = args.length,rescle=args.rescle, arng=args.a,sal=args.sal)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    print('\nTotal adversarial train accuarcy:', 100. * correct / total)
    print('Total adversarial train loss:', train_loss)

def test(epoch):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            benign_loss += loss.item()

            _, predicted = outputs.max(1)
            benign_correct += predicted.eq(targets).sum().item()


            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            adv_loss += loss.item()

            _, predicted = adv_outputs.max(1)
            adv_correct += predicted.eq(targets).sum().item()
    benign = 100. * benign_correct / total
    adv = 100. * adv_correct / total
    print('\nTotal benign test accuarcy:', benign)
    print('Total adversarial test Accuarcy:', adv)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
    print('Model Saved!')
    return benign, adv

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

benignH = 0
advH = 0
benignt = 0
advt = 0
for epoch in range(0, 200):
    adjust_learning_rate(optimizer, epoch)
    train(epoch)
    benign, adv = test(epoch)
    if benign > benignH:
        print('epoch ', epoch,':', 'best benign and adv accuracy: ', benign, ', ', adv)
        benignH = benign
        advt = adv
        epochbe = epoch
    if adv > advH:
        print('epoch ', epoch,':', 'benign and best adv accuracy: ', benign, ', ', adv)
        advH = adv
        benignt = benign
        epochadv = epoch

print('epoch ', epochadv,':', 'benign and adv accuracy for best adversarial: ', benignt, ', ', advH)
print('epoch ', epochbe,':', 'benign and adv accuracy for best natural: ', benignH, ', ', advt)
   
