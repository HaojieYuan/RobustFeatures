import sys
sys.path.append('../')
import cifar10_models
from cifar10_train import udf_dataset
import torch
from attacks import projected_gradient_descent
from tqdm import tqdm
import torchvision.transforms as transforms
import pdb

model_path = '/home/haojieyuan/RobustFeatures/cifar_normal_adam_best.pth.tar'
#model_path = '/home/haojieyuan/RobustFeatures/cifar_high_disentangle_adam_best.pth.tar'

model = cifar10_models.ResNet50()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(model_path)
normal_acc = checkpoint['best_acc1']
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
model.eval()

# Need to normalize after attack
# Thus preprocess is splited.
cifar10_preprocess_2tensor = transforms.Compose([
        transforms.ToTensor()
    ])
cifar10_preprocess_norm = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

normal_dataset = udf_dataset('/home/haojieyuan/Data/CIFAR_10_data/images/val',
                             '/home/haojieyuan/Data/CIFAR_10_data/images/val.txt',
                             transform=cifar10_preprocess_2tensor, frequency='normal')

data_loader = torch.utils.data.DataLoader(
        normal_dataset, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True)


correct = 0
total = 0

for data in tqdm(data_loader):
    imgs, labels = data
    # eps 0.25, step eps 0.05, iter 20, L2 Norm
    adv_imgs = projected_gradient_descent(model, imgs.cuda(), 0.25, 0.05, 20, 2,
                                          clip_min=0., clip_max=1., y=labels.cuda(), targeted=False,
                                          rand_init=True, sanity_checks=True, transform=cifar10_preprocess_norm)
    adv_imgs = adv_imgs.detach()

    with torch.no_grad():
        adv_imgs = torch.stack([cifar10_preprocess_norm(img_adv) for img_adv in adv_imgs])
        outputs = model(adv_imgs.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu()==labels).sum().detach().cpu().item()

print('Normal accuracy: %f %%'%(normal_acc))
print('Adv accuracy: %f %%'%(100.*correct/total))

#pdb.set_trace()
