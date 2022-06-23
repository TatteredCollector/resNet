import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model.resNet import resnet34


def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          # 尽量按比例缩放
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                               std=(0.229, 0.224, 0.225))])
    # 获得路径
    imag_path = './data/images'
    assert os.path.exists(imag_path), "file path {} is not exist!".format(imag_path)
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    #  endswith() 方法用于判断字符串是否以指定后缀结尾，
    img_path_list = [os.path.join(imag_path, i) for i in os.listdir(imag_path) if i.endswith('.jpg')]

    json_path = "./class_dict.json"
    assert os.path.exists(json_path), "file path {} is not exist!".format(json_path)

    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    model = resnet34(num_classes=5)

    model.to(device)
    model_weight_path = "./resnet34.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    model.eval()
    batch_size = 8
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size + 1):
            img_list = []
            for img_path in img_path_list[ids * batch_size:min(batch_size * (ids + 1), len(img_path_list))]:
                img = Image.open(img_path)
                img = data_transforms(img)
                img_list.append(img)
                # 每批次内的每一张图片作为元素的列表

            # 将列表转换成熟悉的 N*C*H*W张量
            batch_img = torch.stack(img_list, dim=0)
            output = model(batch_img.to(device)).cpu()
            predict_y = torch.softmax(output, dim=1)
            prob, classes = torch.max(predict_y, dim=1)
            # zip() 把多个相同数量可迭代的变量组合成一个元组 
            for idx,(pro,cla) in enumerate(zip(prob,classes)):
                print("image :{} class:{} prob:{:.3f}".format(img_path_list[ids*batch_size+idx],
                                                              class_dict[str(cla.numpy())],
                                                              pro.numpy()))


if __name__ == "__main__":
    predict()
