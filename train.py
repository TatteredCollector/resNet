import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils import data
from torch import optim
from tqdm import tqdm
from torchsummary import summary

from model.resNet import resnet34


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    data_transforms = {"train": transforms.Compose([transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                         std=(0.229, 0.224, 0.225))]),
                       "val": transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  # 先按比例缩放到256
                                                  # 再变成224*224的正方形
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                       std=(0.229, 0.224, 0.225))
                                                  ])}
    data_root = os.getcwd()
    image_path = os.path.join(data_root, 'data')
    assert os.path.exists(image_path), "{} data_path not exist!".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transforms["val"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)

    class_dict = train_dataset.class_to_idx
    class_dict = dict((key, val) for val, key in class_dict.items())

    json_str = json.dumps(class_dict, indent=4)
    with open('class_dict.json', 'w') as f:
        f.write(json_str)

    batch_size = 16
    nw = min(os.cpu_count(), batch_size if batch_size > 0 else 0, 8)
    print("using {} dataloader worker every process".format(nw))
    print("using image {} in train, {} in val".format(train_num, val_num))

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=nw)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=nw)
    net = resnet34()

    # 迁移学习 加载别人的权重，每一个卷积命名要与他的高度一致
    model_wight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_wight_path), "path is not exist!".format(model_wight_path)
    net.load_state_dict(torch.load(model_wight_path, map_location=device))
    # 固定前面的所有卷积层参数
    # for param in net.parameters():
    #   param.requires_grad = False
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, out_features=5)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    param = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(param, lr=0.0001)

    epochs = 10
    best_acc = 0.0
    save_path = "./resnet34.pth"
    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()
        all_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, train_data in enumerate(train_bar):
            images, labels = train_data
            out = net(images.to(device))
            loss = loss_function(out, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss += loss.item()

            train_bar.desc = "train epoch {} loss:{:.3f}".format(epoch + 1, loss)

        # val
        net.eval()
        acc = 0.0
        val_bar = tqdm(val_loader)
        with torch.no_grad():
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_out = net(val_images.to(device))
                # dim 0:每列最大值 1 :每行最大值
                predict = torch.max(val_out, dim=1)[1]

                acc += torch.eq(predict, val_labels.to(device)).sum().item()

        val_acc = acc / val_num
        print("epoch {} train loss : {:.3f} val_accuracy:{:.3f}".format(
            epoch + 1, all_loss / train_steps, val_acc
        ))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)

    print("finish")


if __name__ == "__main__":
    train()
