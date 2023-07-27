import argparse
import logging
import os
# from threading import local
import time

import torch
# import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


import losses
from config import config as cfg
from dataset import MXFaceDataset, DataLoaderX, FaceDataset, FaceDataset_2class
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
import torch.utils.data as data

from backbones.iresnet import iresnet100, iresnet50


torch.backends.cudnn.benchmark = True

def main(args):
    # dist.init_process_group(backend='nccl', init_method='env://')
    # torch.cuda.set_device(local_rank)
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    local_rank = args.local_rank
    world_size =1

    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)
    else:
        time.sleep(2)
    rank = 0
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    trainset = FaceDataset_2class(root_dir=cfg.rec, json_path = "/media/thainq97/DATA/GHTK/sonnt373/CR-FIQA_Tuning/stress_test/data_train_2class_2407.json")

    train_loader = data.DataLoader(
            dataset=trainset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=12, pin_memory=True, drop_last=True)
    
    # exit()
    # load evaluation
    print(f"name of network: {cfg.network}")

    if cfg.network == "iresnet100":
        backbone = iresnet100(dropout=0.4,num_features=cfg.embedding_size).cuda()
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=False, qs=1).cuda()
    else:
        backbone = None
        logging.info("load backbone failed!")

    try:
        backbone_pth = "/media/thainq97/DATA/GHTK/sonnt373/CR-FIQA_Tuning/cp/181952backbone.pth"
        backbone.load_state_dict(torch.load(backbone_pth, map_location='cuda'))
        logging.info("backbone student loaded successfully!")
        logging.info("backbone resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("load backbone resume init, failed!")
    backbone.cuda()
    backbone.train()

    ## If Get name of layer of model
    i = 0
    for name, param in backbone.named_parameters():
        print(f'layer num {i} name {name}')
        i += 1
    exit()

    # if Freeze backbone
    i = 0
    for param in backbone.parameters():
        i += 1
        if i < 459: #459 
            param.requires_grad = False
    
    # get header
    if args.loss == "CR_FIQA_LOSS":
        header = losses.CR_FIQA_LOSS(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m)
    else:
        print("Header not implemented")
    try:
        # header_pth = os.path.join(cfg.identity_model, str(cfg.identity_step) + "header.pth")
        header_pth = "/media/thainq97/DATA/GHTK/sonnt373/CR-FIQA_Tuning/output/R50_CRFIQA_resnet100/26172header.pth"
        header.load_state_dict(torch.load(header_pth, map_location=torch.device('cuda')))

        if rank == 0:
            logging.info("header resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError):
        logging.info("header resume init, failed!")

    header.cuda()
    header.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)

    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.lr_func)        

    criterion = CrossEntropyLoss()
    criterion_qs = torch.nn.SmoothL1Loss(beta=0.5)
    start_epoch = 0

    total_step = int(len(trainset) / cfg.batch_size * cfg.num_epoch)
    logging.info("Total Step is: %d" % total_step)

    if args.resume:
        rem_steps = (total_step - cfg.global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

        print("last learning rate: {}".format(scheduler_header.get_lr()))
        # ------------------------------------------------------------

    # callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)
    alpha=10.0  #10.0
    loss = AverageMeter()
    global_step = cfg.global_step

    print("--------------start training---------------")
    for epoch in range(start_epoch, cfg.num_epoch):
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            features, qs = backbone(img)
            qs_sigmoid = torch.nn.Sigmoid()(qs)
            qs_sigmoid = qs_sigmoid.cuda(local_rank, non_blocking=True)
            loss_qssigmoid = F.binary_cross_entropy(qs_sigmoid.squeeze(), label.float())
            loss_qssigmoid.backward()
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            opt_backbone.step()
            opt_backbone.zero_grad()
            loss.update(loss_qssigmoid.item(), 1)
            callback_logging(global_step, loss, epoch, 0,loss_qssigmoid)
            
        print(f"done epoch {epoch}")
        scheduler_backbone.step()

        callback_checkpoint(global_step, backbone, header)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CR-FIQA Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--loss', type=str, default="CR_FIQA_LOSS", help="loss function")
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)
