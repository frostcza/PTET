import os, argparse
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from models.model_cfg import net_for_test
import cfg
from dataset import *
from PIL import Image
import time
from thop import profile

parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=1, help='train batch size')
parser.add_argument('--input-h', type=int, default=480, help='input h')
parser.add_argument('--input-w', type=int, default=640, help='input w')

parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='select gpu.')

parser.add_argument('--profile', type=bool, default=False, help='print profile information.')

params = parser.parse_args()

test_data_dir = "./data/MSRS/test/"
save_dir = "./results/"

cfg.num_parallel = 3
cfg.predictor_num_parallel = 2
cfg.use_exchange = True

# Data pre-processing
transform = transforms.Compose([transforms.Resize((480,640)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Test data
test_data = DatasetFromFolder(test_data_dir, img_type="png", transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=params.batch_size,
                                               shuffle=False, drop_last=False)

# Models
torch.cuda.set_device(params.gpu[0])
Net = net_for_test()
Net.cuda()

state_dict = torch.load('./ckpt/model/checkpoint-119.pkl', map_location='cuda:'+str(params.gpu[0]))
Net.load_state_dict(state_dict, strict=True)


with torch.no_grad():
    time_list = []
    for i, inputs in tqdm(enumerate(test_data_loader), miniters=25, total=len(test_data_loader)):
        x = [p.cuda() for p in inputs]
        
        if params.profile:
            flops, parameters = profile(Net, inputs=(x[0],x[1], ))
            print("FLOPs : ", flops)
            print("Params : ", parameters)
            break
        
        start=time.time()
        
        gen_fused = Net(x[0],x[1])
        
        end=time.time()
        time_list.append(end-start)
        print(" time: ", (end-start))
        
        gen_fused = gen_fused.cpu().data
        img = (((gen_fused[0] - gen_fused[0].min()) * 255) / (gen_fused[0].max() - gen_fused[0].min()))\
            .numpy().transpose(1, 2, 0).astype(np.uint8)
            
        img = Image.fromarray(img)
    
        save_dir_ = os.path.join(save_dir, 'result1')
        if not os.path.exists(save_dir_):
            os.mkdir(save_dir_)
        file_name = os.path.join(save_dir_, '%03d.png' % i)
        img.save(file_name, "PNG")

    if not params.profile:
        del time_list[0]
        print("Average testing time is [%f]"%(np.mean(time_list)))
        print("Min testing time is [%f]"%(np.min(time_list)))
        print("Max testing time is [%f]"%(np.max(time_list)))