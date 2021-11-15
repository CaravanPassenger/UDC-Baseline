import torch
from SubEncoder_Unet import DE_Unet
import torch.optim as optim
from Dataset import TrainDataset, get_train_loader
from torch.autograd import Variable
from Utils import Option
import torch.nn as nn
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np

def train(model, train_loader, criterion, epoch):
    """
    完成一次epoch的训练
    :param model:待训练模型
    :param train_loader:训练数据加载
    :param criterion:误差计算设置
    :param epoch:第几个epoch
    :return:
    """
    # 记录训练了几个batch，并求出每个batch的平均损失
    num_batches = 0
    avg_loss = 0
    # log文件记录训练
    with open(Option.logs_path, 'a') as file:
        for batch_idx, sample_batched in enumerate(train_loader):
            whl_num = epoch
            # 其中sample_batched是标签和数据的集合（两个key），分离
            image = sample_batched['image']
            writer.add_images("input" + str(whl_num), image, batch_idx+1)
            label = sample_batched['label']
            writer.add_images("target" + str(whl_num), label, batch_idx+1)
            # 将图片数据类型统一，设为Variable，并且加载到GPU上
            # image, label = Variable(image.type(torch.float32)).cuda(),\
            #                Variable(label.type(torch.float32)).cuda()
            image,label=image.type(torch.float32).cuda(),label.type(torch.float32).cuda()
            # 梯度初始化
            optimizer.zero_grad()
            # 前向运算
            output = model(image)
            # 计算误差，梯度，反向传播
            loss = criterion(output, label)
            print(loss)
            writer.add_images("output" + str(whl_num), output, batch_idx+1)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            num_batches += 1
        # 计算平均误差
        avg_loss /= num_batches
        # 记录loss
        writer.add_scalar("loss", avg_loss, epoch)
        print('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss))
        file.write('epoch: ' +str(epoch) +', train loss: ' +str(avg_loss) +'\n')
    scheduler.step()
    print(f"第{epoch}个epoch,学习率更新为：{optimizer.state_dict()['param_groups'][0]['lr']}")


def run(model, train_loader, criterion,model_save_path):
    # 训练十次
    #Option.epochs
    for epoch in range(1,200):
        train(model, train_loader, criterion, epoch)
    torch.save(model, model_save_path)


if __name__ == '__main__':
    # 引入tensorboradX
    writer = SummaryWriter()
    # 配置GPU
    device = torch.device('cuda:0')
    # 生成数据读取器
    train_dataloader = get_train_loader(HandLpath=Option.Toled_train_path)
    # 模型实例化并加载在GPU上
    model=DE_Unet()
    # model = torch.load(Option.model_save_path)
    model.to(device)
    # 优化器与误差
    optimizer = optim.Adam(
        model.parameters(),
        lr=Option.learning_rate,
        weight_decay=Option.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400,600], gamma=0.5)
    criterion = nn.L1Loss().cuda()
    # 训练模型
    run(model, train_dataloader, criterion,Option.model_save_path)
