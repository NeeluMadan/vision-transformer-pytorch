import torch
import random
import argparse
import time  
# import utils
# import config
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from metrics import accuracy 
#import models
from model import ViT
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
#from scheduler import WarmupCosineSchedule
from torchvision import datasets, transforms
from dataloader import ImageFolder
from torch.utils.data import DataLoader
import os 


def get_args_parser():
    parser = argparse.ArgumentParser(description='vision transformer')
    parser.add_argument('--data', type=str, default='imagenet', metavar='N',
                        help='data')
    parser.add_argument('--data_path', type=str, default='/home/nelu/MRPA_Projects/datasets/ImgNet_100', metavar='N',
                        help='data') 
    parser.add_argument('--model', type=str, default='vit', metavar='N',
                        help='model')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=0.3, metavar='M',
                        help='adam weight_decay (default: 0.5)')
    parser.add_argument('--t_max', type=float, default=80000, metavar='M',
                        help='cosine annealing steps')                    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--alpha', type=int, default=0.9, metavar='N',
                        help='alpha')
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                        help='num_workers')                                         
    parser.add_argument('--mode',type=str,default='val',
                        help = 'train/val')                    
    ## DDP
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--gpu',type=str,default='0',
                        help = 'gpu')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', type=int, default=1, metavar='N',
                        help='world_size')
    parser.add_argument('--distributed', default=True, type=bool)

    return parser
                   
#os.environ["WANDB_API_KEY"] = ' '
cudnn.benchmark=True 

def main(args) : 
    st = time.time()
    train_dir = os.path.join(args.data_path,'train')
    test_dir = os.path.join(args.data_path,'val')

    trainset = ImageFolder(
        train_dir,
        transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]))
    valset = ImageFolder(
        test_dir,
        transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])) 
    
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    model = ViT(in_channels = 3,
            patch_size = 16,
            emb_size = 768,
            img_size = 256,
            depth = 12,
            n_classes = 100,
            )

    if args.mode == 'train':
        model = model.cuda()

        optimizer = optim.AdamW( model.parameters(), lr=args.lr,weight_decay = args.weight_decay)

        #learning rate warmup of 80k steps
        t_total = ( len(train_loader.dataset) / args.batch_size ) * args.epochs  
        print(f"total_step : {t_total}")

        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=0)
        criterion = nn.CrossEntropyLoss().cuda()
        
        #clip_norm
        max_norm = 1 

        #train 
        scaler = torch.cuda.amp.GradScaler()

        model.train()
        for epoch in range(args.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                st = time.time()
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    mid_x = mid_y = int(data.size()[-1]/2)

                    out1 = model(data[:,:,0:mid_y, 0:mid_x])
                    out2 = model(data[:,:,0:mid_y, mid_x:])
                    out3 = model(data[:,:,mid_y:, 0:mid_x])
                    out4 = model(data[:,:,mid_y:, mid_x:])
                    output = (out1+out2+out3+out4)/2
                    output += 1e-8
                    loss = criterion(output,target) 
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                scaler.step(optimizer)
                scaler.update() 
                scheduler.step()  #iter
                #train_loss = reduce_tensor(loss.data)
                train_loss = loss.item()

                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, train_loss))
                print('teacher_network iter time is {0}s'.format(time.time()-st))
            
            if epoch % 15 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                },filename=f'checkpoints/checkpoint_{epoch}.pth.tar') 
            
            model.eval()
            correct = 0
            total_acc1 = 0
            total_acc5 = 0
            step=0
            st = time.time()
            for batch_idx,(data, target) in enumerate(val_loader) :
                with torch.no_grad() :
                    data, target = data.cuda(), target.cuda()
                    mid_x = mid_y = int(data.size()[-1]/2)

                    out1 = model(data[:,:,0:mid_y, 0:mid_x])
                    out2 = model(data[:,:,0:mid_y, mid_x:])
                    out3 = model(data[:,:,mid_y:, 0:mid_x])
                    out4 = model(data[:,:,mid_y:, mid_x:])
                    output = (out1+out2+out3+out4)/2
                    output += 1e-8

                val_loss = criterion(output,target) 
                #val_loss = reduce_tensor(val_loss.data)
                

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                total_acc1 += acc1[0] 
                total_acc5 += acc5[0]
                step+=1
            
            print('\nval set: top1: {}, top5 : {} '.format(total_acc1/step, total_acc5/step))
            print(f"validation time : {time.time()-st}")

    if args.mode == 'val' : 
        checkpoint = torch.load('./checkpoint_285.pth.tar')
        model = model.cuda()
        model.load_state_dict(checkpoint['state_dict'])

        optimizer = optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.weight_decay) 
        optimizer.load_state_dict(checkpoint['optimizer'])

        criterion = nn.CrossEntropyLoss().cuda() 
    
        model.eval()
        correct = 0
        total_acc1 = 0
        total_acc5 = 0

        st = time.time()
        for batch_idx,(data, target) in enumerate(val_loader) :
            with torch.no_grad() :
                data, target = data.cuda(), target.cuda()
                mid_x = mid_y = int(data.size()[-1]/2)

                out1 = model(data[:,:,0:mid_y, 0:mid_x])
                out2 = model(data[:,:,0:mid_y, mid_x:])
                out3 = model(data[:,:,mid_y:, 0:mid_x])
                out4 = model(data[:,:,mid_y:, mid_x:])
                output = (out1+out2+out3+out4)/2
            val_loss = criterion(output,target) 
            #val_loss = val_loss.item()
            #val_loss = reduce_tensor(val_loss.data)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_acc1 += acc1[0] 
            total_acc5 += acc5[0]

            
        print('\nval set: top1: {}, top5 : {} '.format(torch.mean(total_acc1), torch.mean(total_acc5)))
        
        print(f"validation time : {time.time()-st}")

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)



if __name__ == '__main__' :  
   args = get_args_parser()
   args = args.parse_args()
   main(args)
