from utils.dataloader import TrainImageReader,convert_image_to_tensor,ImageDB
import datetime
import os
from utils.models  import PNet,RNet,ONet,LossFn
import torch
#from torch.autograd import Variable 新版本中已弃用
import utils.config as config
import argparse
import sys
sys.path.append(os.getcwd())
import numpy as np



def compute_accuracy(prob_cls, gt_cls):

    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    #we only need the detection which >= 0
    mask = torch.ge(gt_cls,0)
    #get valid element
    valid_gt_cls = torch.masked_select(gt_cls,mask)
    valid_prob_cls = torch.masked_select(prob_cls,mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls,0.6).float()
    right_ones = torch.eq(prob_ones,valid_gt_cls).float()

    ## if size == 0 meaning that your gt_labels are all negative, landmark or part

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))  ## divided by zero meaning that your gt_labels are all negative, landmark or part


def train_pnet(model_store_path, end_epoch,imdb,
              batch_size,frequent=10,base_lr=0.01,lr_epoch_decay=[9],use_cuda=True,load=''):
    
    #create lr_list
    lr_epoch_decay.append(end_epoch+1)
    lr_list = np.zeros(end_epoch)
    lr_t = base_lr
    for i in range(len(lr_epoch_decay)):
        if i==0:
            lr_list[0:lr_epoch_decay[i]-1]=lr_t
        else:
            lr_list[lr_epoch_decay[i-1]-1:lr_epoch_decay[i]-1]=lr_t
        lr_t*=0.1


    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = PNet(is_train=True, use_cuda=use_cuda)
    if load!='':
        net.load_state_dict(torch.load(load))
        print('model loaded',load)
    net.train()

    if use_cuda:
        net.cuda()
    

    optimizer = torch.optim.Adam(net.parameters(), lr=lr_list[0])
    #optimizer = torch.optim.SGD(net.parameters(), lr=lr_list[0])

    train_data=TrainImageReader(imdb,12,batch_size,shuffle=True)

    #frequent = 10
    for cur_epoch in range(1,end_epoch+1):
        train_data.reset() # shuffle
        for param in optimizer.param_groups:
            param['lr'] = lr_list[cur_epoch-1]
        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):

            im_tensor = [ convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor.requires_grad = True
            gt_label = torch.from_numpy(gt_label).float()
            gt_label.requires_grad = True

            gt_bbox = torch.from_numpy(gt_bbox).float()
            gt_bbox.requires_grad = True
            # gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                # gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*1.0+box_offset_loss*0.5

            if batch_idx %frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                # show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show5,lr_list[cur_epoch-1]))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(model_store_path,"pnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"pnet_epoch_model_%d.pkl" % cur_epoch))




def train_rnet(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,lr_epoch_decay=[9],use_cuda=True,load=''):

    #create lr_list
    lr_epoch_decay.append(end_epoch+1)
    lr_list = np.zeros(end_epoch)
    lr_t = base_lr
    for i in range(len(lr_epoch_decay)):
        if i==0:
            lr_list[0:lr_epoch_decay[i]-1]=lr_t
        else:
            lr_list[lr_epoch_decay[i-1]-1:lr_epoch_decay[i]-1]=lr_t
        lr_t*=0.1
    #print(lr_list)
    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = RNet(is_train=True, use_cuda=use_cuda)
    net.train()
    if load!='':
        net.load_state_dict(torch.load(load))
        print('model loaded',load)
    if use_cuda:
        net.cuda()
    

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,24,batch_size,shuffle=True)


    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        for param in optimizer.param_groups:
            param['lr'] = lr_list[cur_epoch-1]

        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):

            im_tensor = [ convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor.requires_grad = True
            gt_label = torch.from_numpy(gt_label).float()
            gt_label.requires_grad = True

            gt_bbox = torch.from_numpy(gt_bbox).float()
            gt_bbox.requires_grad = True
            gt_landmark = torch.from_numpy(gt_landmark).float()
            gt_landmark.requires_grad = True

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*1.0+box_offset_loss*0.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                # show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(), cur_epoch, batch_idx, show1, show2, show3, show5, lr_list[cur_epoch-1]))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(model_store_path,"rnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"rnet_epoch_model_%d.pkl" % cur_epoch))


def train_onet(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,lr_epoch_decay=[9],use_cuda=True,load=''):
    #create lr_list
    lr_epoch_decay.append(end_epoch+1)
    lr_list = np.zeros(end_epoch)
    lr_t = base_lr
    for i in range(len(lr_epoch_decay)):
        if i==0:
            lr_list[0:lr_epoch_decay[i]-1]=lr_t
        else:
            lr_list[lr_epoch_decay[i-1]-1:lr_epoch_decay[i]-1]=lr_t
        lr_t*=0.1
    #print(lr_list)

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = ONet(is_train=True)
    if load!='':
        net.load_state_dict(torch.load(load))
        print('model loaded',load)
    net.train()
    #print(use_cuda)
    if use_cuda:
        net.cuda()
    

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,48,batch_size,shuffle=True)


    for cur_epoch in range(1,end_epoch+1):

        train_data.reset()
        for param in optimizer.param_groups:
            param['lr'] = lr_list[cur_epoch-1]
        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):
            # print("batch id {0}".format(batch_idx))
            im_tensor = [ convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor.requires_grad = True
            gt_label = torch.from_numpy(gt_label).float()
            gt_label.requires_grad = True

            gt_bbox = torch.from_numpy(gt_bbox).float()
            gt_bbox.requires_grad = True
            gt_landmark = torch.from_numpy(gt_landmark).float()
            gt_landmark.requires_grad = True

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred, landmark_offset_pred = net(im_tensor)

            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*0.8+box_offset_loss*0.6+landmark_loss*1.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, landmark loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show4,show5,base_lr))
                #print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show5,lr_list[cur_epoch-1]))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(model_store_path,"onet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))






def parse_args():
    parser = argparse.ArgumentParser(description='Train MTCNN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--net', dest='net', help='which net to train', type=str)

    parser.add_argument('--anno_file', dest='annotation_file', help='training data annotation file', type=str)
    parser.add_argument('--model_path', dest='model_store_path', help='training model store directory',
                        default=config.MODEL_STORE_DIR, type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=config.END_EPOCH, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=config.TRAIN_LR, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=config.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=config.USE_CUDA, type=bool)
    parser.add_argument('--load', dest='load', help='load model', type=str)

    args = parser.parse_args()
    return args

def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01,lr_epoch_decay=[9],
                 batch_size=128, use_cuda=False,load='',net='pnet'):
    if net=='pnet':
        annotation_file = os.path.join(config.ANNO_STORE_DIR,config.PNET_TRAIN_IMGLIST_FILENAME)
    elif net=='rnet':
        annotation_file = os.path.join(config.ANNO_STORE_DIR,config.RNET_TRAIN_IMGLIST_FILENAME)
    elif net=='onet':
        annotation_file = os.path.join(config.ANNO_STORE_DIR,config.ONET_TRAIN_IMGLIST_FILENAME)
    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    print('DATASIZE',len(gt_imdb))
    gt_imdb = imagedb.append_flipped_images(gt_imdb)
    print('FLIP DATASIZE',len(gt_imdb))
    if net=="pnet":
        print("Training Pnet:")
        train_pnet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr,lr_epoch_decay=lr_epoch_decay, use_cuda=use_cuda,load=load)
    elif net=="rnet":
        print("Training Rnet:")
        train_rnet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr,lr_epoch_decay=lr_epoch_decay, use_cuda=use_cuda,load=load)
    elif net=="onet":
        print("Training Onet:")
        train_onet(model_store_path=model_store_path, end_epoch=end_epoch, imdb=gt_imdb, batch_size=batch_size, frequent=frequent, base_lr=lr,lr_epoch_decay=lr_epoch_decay, use_cuda=use_cuda,load=load)

if __name__ == '__main__':
    args = parse_args()
    lr_epoch_decay = [9]
    train_net(annotation_file=args.annotation_file, model_store_path=args.model_store_path,
                end_epoch=args.end_epoch, frequent=args.frequent, lr=args.lr,lr_epoch_decay=lr_epoch_decay,batch_size=args.batch_size, use_cuda=args.use_cuda,load=args.load,net=args.net)