import os
import time
import datetime
import torch
import torch.nn as nn

from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from src.ConvUNeXt import ConvUNeXt as UNet

transform=transforms.Compose([
        transforms.RandomResizedCrop(480),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
glioma_train = datasets.ImageFolder("archive/Training", transform=transform)
glioma_test = datasets.ImageFolder("archive/Testing", transform=transform)

train_loader = torch.utils.data.DataLoader(glioma_train, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(glioma_test, batch_size=16, shuffle=True)


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        num = -1
        for imgs, labels in train_loader:
            num = num+1
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            # print('{} Epoch[{}], [{}/200] loss '.format(
            #     datetime.datetime.now(), epoch, num
            # ))

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)
        ))

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

model = create_model(num_classes=4)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4,weight_decay=5e-5)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader = train_loader,
)

def validate(model, train_loader, test_loader):
    for name, loader in [("train", train_loader), ("test", test_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        print("Accuracy {}: {:.2f}".format(name , correct / total))
        print(total)

validate(model, train_loader, test_loader)
# def main(args):
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     batch_size = args.batch_size
#
#     # 用来保存训练以及验证过程中信息
#     results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#
#     train_loader = torch.utils.data.DataLoader(glioma_train, batch_size=1, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(glioma_test, batch_size=1, shuffle=True)
#
#     num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
#
#     model = ViT(batchsize=1,
#                 img_size=224,
#                 patch_size=16,
#                 head_num=12,
#                 embed_dim=768,
#                 dropout=0.1,
#                 num_classes=4)
#     model = model.to(device)
#
#     params_to_optimize = [p for p in model.parameters() if p.requires_grad]
#
#     optimizer = torch.optim.Adam(
#         params_to_optimize,
#         lr=0.0005
#     )
#
#     scaler = torch.cuda.amp.GradScaler() if args.amp else None
#
#     # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
#     lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
#
#     if args.resume:
#         checkpoint = torch.load(args.resume, map_location='cpu')
#         model.load_state_dict(checkpoint['model'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#         args.start_epoch = checkpoint['epoch'] + 1
#         if args.amp:
#             scaler.load_state_dict(checkpoint["scaler"])
#
#     best_dice = 0.
#     start_time = time.time()
#     for epoch in range(args.start_epoch, args.epochs):
#         mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, 4,
#                                         lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
#
#         confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
#         val_info = str(confmat)
#         print(val_info)
#         print(f"dice coefficient: {dice:.3f}")
#         # write into txt
#         with open(results_file, "a") as f:
#             # 记录每个epoch对应的train_loss、lr以及验证集各指标
#             train_info = f"[epoch: {epoch}]\n" \
#                          f"train_loss: {mean_loss:.4f}\n" \
#                          f"lr: {lr:.6f}\n" \
#                          f"dice coefficient: {dice:.3f}\n"
#             f.write(train_info + val_info + "\n\n")
#
#         if args.save_best is True:
#             if best_dice < dice:
#                 best_dice = dice
#             else:
#                 continue
#
#         save_file = {"model": model.state_dict(),
#                      "optimizer": optimizer.state_dict(),
#                      "lr_scheduler": lr_scheduler.state_dict(),
#                      "epoch": epoch,
#                      "args": args}
#         if args.amp:
#             save_file["scaler"] = scaler.state_dict()
#
#         if args.save_best is True:
#             torch.save(save_file, "save_weights/best_model_GlaS.pth")
#         else:
#             torch.save(save_file, "save_weights/model_{}.pth".format(epoch))
#
#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print("training time {}".format(total_time_str))
#
# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser(description="pytorch training")
#
#     parser.add_argument("--device", default="cuda", help="training device")
#     parser.add_argument("-b", "--batch-size", default=1, type=int)
#     parser.add_argument("--epochs", default=1, type=int, metavar="N",
#                         help="number of total epochs to train")
#
#     parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
#     parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
#                         metavar='W', help='weight decay (default: 1e-4)',
#                         dest='weight_decay')
#     parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
#     parser.add_argument('--resume', default='', help='resume from checkpoint')
#     parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                         help='start epoch')
#     parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
#     # Mixed precision training parameters
#     parser.add_argument("--amp", default=False, type=bool,
#                         help="Use torch.cuda.amp for mixed precision training")
#
#     args = parser.parse_args()
#
#     return args
# if __name__ == '__main__':
#     args = parse_args()
#
#     if not os.path.exists("./save_weights"):
#         os.mkdir("./save_weights")
#
#     main(args)

