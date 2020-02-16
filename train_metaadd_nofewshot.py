import argparse
import os.path as osp
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cub200_metaadd import CUB200 as MiniImageNet
# from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from proto_utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-epoch', type=int, default=10) #increase the training set meta-epoch times
    parser.add_argument('--max-epoch', type=int, default=1) #fixing the size of training set
    parser.add_argument('--save-epoch', type=int, default=2) #save model every meta-epoch
    parser.add_argument('--shot', type=int, default=1) #shot is the parameter meta-learner will increase each meta-epoch
    parser.add_argument('--query', type=int, default=10)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/meta_proto-1')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)


    model = Convnet().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
        
        
#     class CategoriesSampler():

#         def __init__(self,indices,batch_size,n_batch):
#             self.indices=indices
#             self.batch_size=batch_size
#             self.n_batch=n_batch
#         def __len__(self):
#             return self.n_batch

#         def __iter__(self):
#             for i_batch in range(n_batch):
#                 self.i=self.i+self.batch_size
#                 return 
        
    def select_least_likely(dataset, label):
        hardest=[]
        for i in range(num_cls):
            least_likely=0
            least_likely_ind=-1
            indices = np.argwhere(label == i).reshape(-1)
            import pdb;pdb.set_trace()
            one_hot=torch.zeros(len(indices),num_cls).scatter_(1, i,1).cuda()
            for ind in indices:
                img=dataset[ind]
                logits=model(imgs)
                if least_likely < abs(logits-one_hot)[i]:
                    least_likely = abs(logits-one_hot)[i]
                    least_likely_ind=ind
            
            hardest.append(least_likely_ind)
        return hardest
                                
    
    
    
    entire_trainset = MiniImageNet('train')

    entire_train_loader = DataLoader(dataset=entire_trainset, shuffle=False, drop_last=True,
                                  num_workers=8, pin_memory=True)
        
    entire_valset = MiniImageNet('val')
    entire_val_loader = DataLoader(dataset=entire_valset, shuffle=False, drop_last=True,
                            num_workers=8, pin_memory=True)

    num_cls=entire_trainset.num_cls()
    
    for meta_epoch in range(1, args.meta_epoch+1):
        if meta_epoch==1:
            hardest_ind=np.random.randint(0, len(entire_trainset), size=100)
            trainset=MiniImageNet('train', allowed_ind=hardest_ind)
        else:
            trainset = MiniImageNet('train', allowed_ind=hardest_ind)
        
        train_loader = DataLoader(dataset=trainset, shuffle=False, drop_last=True,
                                  num_workers=8, pin_memory=True)
        del hardest_ind
        
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0

        timer = Timer()
    
        
        for epoch in range(1, args.max_epoch + 1):
            lr_scheduler.step()

            model.train()

            tl = Averager()
            ta = Averager()

            for i, (imgs,label) in enumerate(train_loader, 1):
                imgs, label = imgs.cuda(), label.cuda()

                logits = model(imgs)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                      .format(epoch, i, len(train_loader), loss.item(), acc))

                tl.add(loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                proto = None; logits = None; loss = None
    #             if i==1:
    #                 break

            tl = tl.item()
            ta = ta.item()

            
            model.eval()

            vl = Averager()
            va = Averager()

            for i, (imgs, label) in enumerate(entire_val_loader, 1):
                imgs, label = imgs.cuda(), label.cuda()

                logits = model(imgs)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)

                vl.add(loss.item())
                va.add(acc)

                proto = None; logits = None; loss = None

            vl = vl.item()
            va = va.item()
            print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

            if va > trlog['max_acc']:
                trlog['max_acc'] = va
                save_model('max-acc')

            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)

        if meta_epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(meta_epoch))
        
        torch.save(trlog, osp.join(args.save_path, 'trlog_'+str(meta_epoch)))
        print('ETA:{}/{}'.format(timer.measure(), timer.measure(meta_epoch / args.meta_epoch)))
        
        
        hardest_ind=select_least_likely(entire_trainset,label)
            
            

