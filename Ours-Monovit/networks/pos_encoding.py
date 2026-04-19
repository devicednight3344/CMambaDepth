import torch
import math
import torch.nn as nn
from PIL import Image
import os
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
def pil_loader_gray(path):
    img = Image.open(path)
    img = img.convert("L")
    return img
class PositionalEncoding(nn.Module):
    def __init__(self, H, W, d_model):
        super(PositionalEncoding, self).__init__()
        # 计算位置编码并将其存储在pe张量中
        if d_model % 2 != 0:
            d_model += 1  # 如果d_model是奇数，调整为偶数
            num = 1
        else:
            num = 0
        self.pe = torch.zeros(H * W, d_model, requires_grad=False, device=torch.device("cuda"))
        self.position = torch.arange(0, H * W, dtype=torch.float, device=torch.device("cuda")).unsqueeze(1) #尺寸为(H*W)x1
        # 计算div_term，用于缩放不同位置的正弦和余弦函数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to("cuda")
        # 使用正弦和余弦函数生成位置编码，对于d_model的偶数索引，使用正弦函数；对于奇数索引，使用余弦函数
        self.pe[:, 0::2] = torch.sin(self.position * div_term)
        self.pe[:, 1::2] = torch.cos(self.position * div_term)
        if num == 1:
            d_model = d_model - 1
        self.pe = self.pe[:, :d_model]
        self.pe = self.pe.reshape(H, W, d_model)
        self.pe = self.pe.transpose(1, 2).transpose(0, 1).unsqueeze(0)# self.pe的尺寸为1xd_modelxHxW
    # 定义前向传播函数
    def forward(self, edge_map, H, W):
        img_edge_Pos_Encoding = edge_map * self.pe# edge_map的尺寸为1x1xHxW
        # x的尺寸为1xd_modelxHxW
        img_edge_Pos_Encoding = F.interpolate(img_edge_Pos_Encoding, size=(H, W), mode='bilinear', align_corners=False)
        return img_edge_Pos_Encoding
if __name__ == '__main__':
    path = "0000000001_test.png"
    img_edge = pil_loader_gray(os.path.join("picture", path))
    resize = transforms.Resize((192, 640), interpolation=Image.ANTIALIAS)
    to_tensor = transforms.ToTensor()
    # img = resize(img)
    # img = np.array(img)
    # img = cv.GaussianBlur(img, (3, 3), 5, 5)
    img_edge = to_tensor(img_edge).unsqueeze(0)
    save_image(img_edge, "Save_picture/smooth_loss_map/0000000001_test.png")
    Pos_Encoding = PositionalEncoding(192,640,3)
    img_edge_Pos_Encoding = Pos_Encoding(img_edge, 48, 160)
    save_image(img_edge_Pos_Encoding, "Save_picture/smooth_loss_map/0000000001_test_pp.png")
    print(img_edge_Pos_Encoding.shape)
    img = torch.ones(2,3,96,320)
    img = img + img_edge_Pos_Encoding