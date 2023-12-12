import os, argparse
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from models.model_cfg import net
import cfg
from dataset import DatasetFromFolder

parser = argparse.ArgumentParser()

# 训练参数设置
parser.add_argument('--batch-size', type=int, default=1, help='train batch size')
parser.add_argument('--num-epochs', type=int, default=120, help='number of train epochs')

parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='select gpu.')
parser.add_argument('--ckpt', default='model0', type=str, metavar='PATH', help='path to save checkpoint (default: model0)')
parser.add_argument('--print-loss', action='store_true', default=True, help='whether print losses during training')

# DatasetFromFolder的参数
parser.add_argument('--input-h', type=int, default=480, help='input h')
parser.add_argument('--input-w', type=int, default=640, help='input w')

# loss function平衡参数
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--alpha', type=float, default=10.0, help='alpha for content loss')
parser.add_argument('--beta', type=float, default=6.0, help='beta for rank loss')
parser.add_argument('--gama', type=float, default=1.0, help='gama for exchange loss')
parser.add_argument('--lamda', type=float, default=0.2, help='lamda for grad loss.')

params = parser.parse_args()

# Directories for loading data and saving results
train_data_dir = "./data/MSRS/train/"
# test_data_dir = "./data/MSRS/test/"
model_dir = os.path.join('ckpt', params.ckpt)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

cfg.num_parallel = 3
cfg.predictor_num_parallel = 2
cfg.use_exchange = True

# Data pre-processing
transform = transforms.Compose([transforms.Resize((params.input_h, params.input_w)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
# Train data
train_data = DatasetFromFolder(train_data_dir, img_type="png", transform=transform)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=params.batch_size,
                                                shuffle=True, drop_last=False)

# Models
torch.cuda.set_device(params.gpu[0])
Net = net()
Net.cuda()

# Loss function
L1_loss = torch.nn.L1Loss().cuda()

laplace = torch.tensor([[0.0, 1.0, 0.0],
                        [1.0, -4.0, 1.0],
                        [0.0, 1.0, 0.0]], dtype=torch.float, requires_grad=False).view(1, 1, 3, 3)
laplace = laplace.cuda()

def img_rank(im):    
    input_h = params.input_h
    input_w = params.input_w
    psize = 160

    im = im.chunk(int(input_h/psize), dim=2)
    im = [im_.chunk(int(input_w/psize), dim=3) for im_ in im]
    ranks_ = []
    for im0 in im:
        ranks = []
        for im1 in im0:
            im1 = im1.reshape(1,3,1,psize*psize)
            idx = im1.argsort(dim=-1, descending=False)
            r = torch.arange(0, psize*psize, dtype=torch.int64).unsqueeze(0)
            r = r.repeat(3,1,1).unsqueeze(0).cuda()
            rank = torch.zeros(1,3,1,psize*psize, dtype=idx.dtype).cuda()
            rank = rank.scatter_(3, idx, r).cuda()
            rank = rank.reshape(1,3,psize,psize)
            ranks.append(rank)
        ranks = torch.cat(ranks, dim=-1)
        ranks_.append(ranks)
    ranks_ = torch.cat(ranks_, dim=-2).cuda()
    ranks_ = ranks_.type(torch.float)
    ranks_ = ranks_ / (psize*psize)
    # print(ranks_)
    
    return ranks_

# Optimizers
optimizer = torch.optim.Adam(Net.parameters(), lr=params.lr * params.batch_size, betas=(0.5, 0.999))

# Training
step = 0

for epoch in range(params.num_epochs):
    # training
    for i, inputs in tqdm(enumerate(train_data_loader), miniters=25, total=len(train_data_loader)):
        x = [input.cuda() for input in inputs]

        gen_IR, gen_VI, gen_Fused = Net(x[0],x[1])
        
        gen_images = [gen_IR, gen_VI, gen_Fused]
        
        # intensity loss
        x_stack = torch.stack((x[0],x[1]),dim=0)
        intensity_loss = L1_loss(gen_images[2], torch.max(x_stack, 0)[0])
        
        # grad loss
        grad_vi = torch.nn.functional.conv2d(x[1], laplace.repeat(3, 3, 1, 1), stride=1, padding=1)
        grad_ir = torch.nn.functional.conv2d(x[0], laplace.repeat(3, 3, 1, 1), stride=1, padding=1)
        grad_gen_image = torch.nn.functional.conv2d(gen_images[2], laplace.repeat(3, 3, 1, 1), stride=1, padding=1)
        
        grad_stack = torch.stack((grad_vi, grad_ir), dim=0)
        grad_loss = L1_loss(grad_gen_image, torch.max(grad_stack, 0)[0])
        
        # rank loss
        rank_ir = img_rank(x[0])
        rank_gen_image = img_rank(gen_images[2])
        rank_loss = L1_loss(rank_gen_image, rank_ir)
        
        # exchange loss
        exchange_loss = L1_loss(gen_images[0], x[1]) + L1_loss(gen_images[1], x[0])
        
        # total loss
        content_loss = intensity_loss + params.lamda * grad_loss
        loss = params.alpha * content_loss + params.beta * rank_loss + params.gama * exchange_loss
        
        # Back propagation
        Net.zero_grad()
        loss.backward()
        optimizer.step()

        if params.print_loss and step % 30 == 0:
            print('Epoch [%d/%d], Step [%d/%d], ' % \
                (epoch + 1, params.num_epochs, i + 1, len(train_data_loader)), end='')
            print('loss: %.4f  rank_loss: %.4f  content_loss %.4f  exchange_loss %.4f' % 
                  (loss.item(), rank_loss, content_loss, exchange_loss), flush=True)

        step += 1
        # break

    if (epoch + 1) % 20 == 0:
        torch.save(Net.state_dict(), os.path.join(model_dir, 'checkpoint-%d.pkl' % epoch))


torch.save(Net.state_dict(), os.path.join(model_dir, 'checkpoint.pkl'))
